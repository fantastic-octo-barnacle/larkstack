//! Lark OAuth login for the admin console.
//!
//! Replaces a static bearer token with a Lark/Feishu OAuth 2.0 authorization-
//! code flow (PKCE) plus a signed session cookie. The OAuth client is one of
//! the entries in the `[lark-apps]` registry, bound via
//! `[console] lark_app = "<name>"`.
//!
//! Bootstrapping: until a Lark app is bound the console is OPEN (mirrors an
//! unset token) so a fresh install can reach its own setup UI. Once bound,
//! every `/api/*` request needs a valid session; a non-empty `[console] admins`
//! list then restricts which Lark accounts may sign in (empty = any tenant
//! user, with a warning).

use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::{
    Json,
    extract::{Query, Request, State},
    http::{HeaderMap, StatusCode, header},
    middleware::Next,
    response::{IntoResponse, Redirect, Response},
};
use axum_extra::extract::cookie::{Cookie, Key, SameSite, SignedCookieJar};
use base64::Engine;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use larkstack_core::{LarkRegistry, default_base_url};
use rand::RngCore;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256, Sha512};
use tracing::{info, warn};
use utoipa::{IntoParams, ToSchema};

use crate::HostState;

const SESSION_COOKIE: &str = "lk_session";
const OAUTH_COOKIE: &str = "lk_oauth";
/// Server-enforced session lifetime; the cookie itself is browser-session
/// scoped, so this is the upper bound.
const SESSION_TTL_SECS: u64 = 7 * 24 * 3600;

// ---- config resolution -----------------------------------------------------

#[derive(Default, Deserialize)]
struct ConsoleSection {
    #[serde(default)]
    lark_app: Option<String>,
    #[serde(default)]
    admins: Vec<String>,
    #[serde(default)]
    redirect_uri: Option<String>,
    #[serde(default)]
    scope: Option<String>,
}

#[derive(Deserialize)]
struct ConfigView {
    #[serde(default)]
    console: ConsoleSection,
    #[serde(default, rename = "lark-apps")]
    lark_apps: LarkRegistry,
}

/// OAuth settings resolved from the live config. `None` => not configured, so
/// the console stays open.
pub struct OAuthConfig {
    app_id: String,
    app_secret: String,
    /// `open.larksuite.com` host — token exchange + user_info.
    open_base: String,
    /// `accounts.larksuite.com` host — the user-facing authorize page.
    accounts_base: String,
    /// Lowercased allowlist; empty = any tenant user.
    admins: Vec<String>,
    redirect_uri: Option<String>,
    scope: Option<String>,
}

/// Resolve `[console]` + its bound `[lark-apps.<name>]` from the live TOML.
pub fn resolve(config_toml: &str) -> Option<OAuthConfig> {
    let view: ConfigView = toml::from_str(config_toml).ok()?;
    let name = view.console.lark_app.as_deref()?;
    let app = view.lark_apps.get(name)?;
    if app.app_id.is_empty() || app.app_secret.is_empty() {
        return None;
    }
    let open_base = if app.base_url.is_empty() {
        default_base_url()
    } else {
        app.base_url.trim_end_matches('/').to_string()
    };
    // open.larksuite.com -> accounts.larksuite.com ; open.feishu.cn -> accounts.feishu.cn
    let accounts_base = open_base.replacen("://open.", "://accounts.", 1);
    Some(OAuthConfig {
        app_id: app.app_id.clone(),
        app_secret: app.app_secret.clone(),
        open_base,
        accounts_base,
        admins: view
            .console
            .admins
            .iter()
            .map(|a| a.trim().to_lowercase())
            .filter(|a| !a.is_empty())
            .collect(),
        redirect_uri: view.console.redirect_uri.clone(),
        scope: view.console.scope.clone(),
    })
}

/// Every sensitive `user_info` identity field, with the contact scope each one
/// needs. The console matches its admin allowlist on `email`/`enterprise_email`,
/// and those fields are only populated when their scope is both granted to the
/// app *and* requested at authorize time — so we request the full identity set
/// by default. Every scope here must also be granted on the Lark app's
/// Permission Management page, or the authorize page fails with error 20027.
const DEFAULT_USER_INFO_SCOPES: &str = concat!(
    "contact:user.email:readonly ",       // email
    "contact:user.employee:readonly ",    // enterprise_email + employee_no
    "contact:user.employee_id:readonly ", // user_id
    "contact:user.phone:readonly",        // mobile
);

/// The `scope` to send to the authorize endpoint. An explicit `[console].scope`
/// wins verbatim (set it empty to request none); otherwise we default to the
/// full identity set when an admin allowlist is configured, and to nothing when
/// it isn't — the open/bootstrap case needs no email and shouldn't force the
/// operator to grant contact permissions just to sign in.
fn effective_scope(cfg: &OAuthConfig) -> Option<String> {
    match cfg.scope.as_deref() {
        Some(s) => {
            let s = s.trim();
            (!s.is_empty()).then(|| s.to_string())
        }
        None if !cfg.admins.is_empty() => Some(DEFAULT_USER_INFO_SCOPES.to_string()),
        None => None,
    }
}

/// The callback URL Lark redirects back to. Prefer the explicit config value
/// (it must match what's registered in the Lark app), else derive from the
/// request host.
fn redirect_uri(cfg: &OAuthConfig, headers: &HeaderMap) -> String {
    if let Some(uri) = cfg.redirect_uri.as_deref().filter(|u| !u.is_empty()) {
        return uri.to_string();
    }
    let host = headers
        .get(header::HOST)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("localhost:8080");
    let local = host.starts_with("localhost") || host.starts_with("127.0.0.1");
    let scheme = headers
        .get("x-forwarded-proto")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.split(',').next())
        .map(str::trim)
        .unwrap_or(if local { "http" } else { "https" });
    format!("{scheme}://{host}/auth/callback")
}

// ---- session + handshake claims --------------------------------------------

#[derive(Serialize, Deserialize)]
struct Session {
    email: String,
    name: String,
    exp: u64,
}

#[derive(Serialize, Deserialize)]
struct Handshake {
    state: String,
    verifier: String,
    /// Pinned so login and callback present Lark the identical redirect_uri.
    redirect_uri: String,
}

fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn rand_token(bytes: usize) -> String {
    let mut buf = vec![0u8; bytes];
    rand::thread_rng().fill_bytes(&mut buf);
    URL_SAFE_NO_PAD.encode(&buf)
}

fn pkce_challenge(verifier: &str) -> String {
    URL_SAFE_NO_PAD.encode(Sha256::digest(verifier.as_bytes()))
}

fn read_session(jar: &SignedCookieJar) -> Option<Session> {
    let cookie = jar.get(SESSION_COOKIE)?;
    let session: Session = serde_json::from_str(cookie.value()).ok()?;
    (session.exp >= now()).then_some(session)
}

// ---- handlers --------------------------------------------------------------

/// `GET /auth/login` — start the OAuth handshake; redirect to Lark.
#[utoipa::path(
    get, path = "/login", tag = "auth",
    responses(
        (status = 303, description = "Redirect to Lark's authorize page"),
        (status = 400, description = "OAuth is not configured"),
        (status = 500, description = "Could not build the authorize URL"),
    ),
)]
pub async fn login(
    State(s): State<HostState>,
    jar: SignedCookieJar,
    headers: HeaderMap,
) -> Response {
    let Some(cfg) = resolve(&s.control.config()) else {
        return (StatusCode::BAD_REQUEST, "Lark OAuth is not configured").into_response();
    };
    let redirect = redirect_uri(&cfg, &headers);
    let state = rand_token(24);
    let verifier = rand_token(48);
    let challenge = pkce_challenge(&verifier);

    let handshake = serde_json::to_string(&Handshake {
        state: state.clone(),
        verifier,
        redirect_uri: redirect.clone(),
    })
    .unwrap_or_default();
    let jar = jar.add(
        Cookie::build((OAUTH_COOKIE, handshake))
            .path("/auth")
            .http_only(true)
            .same_site(SameSite::Lax)
            .secure(redirect.starts_with("https://"))
            .build(),
    );

    let scope = effective_scope(&cfg);
    let mut params = vec![
        ("client_id", cfg.app_id.as_str()),
        ("redirect_uri", redirect.as_str()),
        ("response_type", "code"),
        ("state", state.as_str()),
        ("code_challenge", challenge.as_str()),
        ("code_challenge_method", "S256"),
    ];
    if let Some(scope) = scope.as_deref() {
        params.push(("scope", scope));
    }
    let authorize = format!("{}/open-apis/authen/v1/authorize", cfg.accounts_base);
    match reqwest::Url::parse_with_params(&authorize, &params) {
        Ok(url) => (jar, Redirect::to(url.as_str())).into_response(),
        Err(e) => {
            warn!("building authorize URL failed: {e}");
            (StatusCode::INTERNAL_SERVER_ERROR, "bad authorize URL").into_response()
        }
    }
}

#[derive(Deserialize, IntoParams)]
#[into_params(parameter_in = Query)]
pub struct CallbackQuery {
    /// Authorization code from Lark (present on success).
    code: Option<String>,
    /// Opaque state echoed back; must match the login handshake.
    state: Option<String>,
    /// Error code from Lark (present when the user denied access).
    error: Option<String>,
}

/// `GET /auth/callback` — exchange the code, check the allowlist, mint a
/// session.
#[utoipa::path(
    get, path = "/callback", tag = "auth",
    params(CallbackQuery),
    responses(
        (status = 303, description = "Signed in; redirect to the console"),
        (status = 400, description = "Missing/mismatched login state, or a Lark error"),
        (status = 403, description = "Account is not in [console].admins"),
        (status = 502, description = "Token exchange or user_info request failed"),
    ),
)]
pub async fn callback(
    State(s): State<HostState>,
    jar: SignedCookieJar,
    Query(q): Query<CallbackQuery>,
) -> Response {
    let Some(cookie) = jar.get(OAUTH_COOKIE) else {
        return (StatusCode::BAD_REQUEST, "login state missing or expired").into_response();
    };
    // Consume the handshake cookie regardless of outcome.
    let jar = jar.remove(Cookie::build((OAUTH_COOKIE, "")).path("/auth").build());
    let Ok(hs) = serde_json::from_str::<Handshake>(cookie.value()) else {
        return (jar, (StatusCode::BAD_REQUEST, "bad login state")).into_response();
    };

    if let Some(err) = q.error {
        return (jar, (StatusCode::BAD_REQUEST, format!("Lark error: {err}"))).into_response();
    }
    let (Some(code), Some(state)) = (q.code, q.state) else {
        return (jar, (StatusCode::BAD_REQUEST, "missing code or state")).into_response();
    };
    if state != hs.state {
        return (jar, (StatusCode::BAD_REQUEST, "state mismatch")).into_response();
    }
    let Some(cfg) = resolve(&s.control.config()) else {
        return (
            jar,
            (StatusCode::BAD_REQUEST, "Lark OAuth is not configured"),
        )
            .into_response();
    };

    let token = match exchange_code(&s.http, &cfg, &code, &hs.verifier, &hs.redirect_uri).await {
        Ok(t) => t,
        Err(e) => {
            warn!("oauth token exchange failed: {e}");
            return (jar, (StatusCode::BAD_GATEWAY, "token exchange failed")).into_response();
        }
    };
    let user = match fetch_user(&s.http, &cfg, &token).await {
        Ok(u) => u,
        Err(e) => {
            warn!("oauth user_info failed: {e}");
            return (
                jar,
                (StatusCode::BAD_GATEWAY, "could not fetch Lark user info"),
            )
                .into_response();
        }
    };

    let emails = user.emails();
    let open_id_ok = user
        .open_id
        .as_deref()
        .is_some_and(|id| cfg.admins.iter().any(|a| a == id));
    if !cfg.admins.is_empty() && !open_id_ok && !emails.iter().any(|e| cfg.admins.contains(e)) {
        if emails.is_empty() {
            warn!(
                "denied console login: Lark returned no email for this account. Grant the app \
                 contact:user.email:readonly (and contact:user.employee:readonly), and make sure \
                 the account has an email set — otherwise no email can match [console].admins."
            );
        } else {
            warn!("denied console login for {emails:?} — not in [console].admins");
        }
        if let Some(id) = user.open_id.as_deref() {
            warn!("denied account open_id={id} (usable in [console].admins)");
        }
        return (
            jar,
            (
                StatusCode::FORBIDDEN,
                "This Lark account is not allowed to access the console.",
            ),
        )
            .into_response();
    }
    let identity = emails
        .first()
        .cloned()
        .or_else(|| user.open_id.clone())
        .unwrap_or_default();
    if cfg.admins.is_empty() {
        warn!("[console].admins is empty — any tenant user can sign in ({identity})");
    }

    let session = serde_json::to_string(&Session {
        email: identity,
        name: user.name.unwrap_or_default(),
        exp: now() + SESSION_TTL_SECS,
    })
    .unwrap_or_default();
    let jar = jar.add(
        Cookie::build((SESSION_COOKIE, session))
            .path("/")
            .http_only(true)
            .same_site(SameSite::Lax)
            .secure(hs.redirect_uri.starts_with("https://"))
            .build(),
    );
    (jar, Redirect::to("/")).into_response()
}

/// `POST /auth/logout` — drop the session cookie.
#[utoipa::path(
    post, path = "/logout", tag = "auth",
    responses((status = 204, description = "Session cleared")),
)]
pub async fn logout(jar: SignedCookieJar) -> Response {
    let jar = jar.remove(Cookie::build((SESSION_COOKIE, "")).path("/").build());
    (jar, StatusCode::NO_CONTENT).into_response()
}

#[derive(Serialize, ToSchema)]
pub struct MeUser {
    email: String,
    name: String,
}

#[derive(Serialize, ToSchema)]
pub struct Me {
    /// Whether OAuth is configured at all. When false the console is open.
    auth_required: bool,
    authenticated: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    user: Option<MeUser>,
}

/// `GET /auth/me` — ungated; tells the UI whether to show the login screen.
#[utoipa::path(
    get, path = "/me", tag = "auth",
    responses((status = 200, description = "Auth + current session state", body = Me)),
)]
pub async fn me(State(s): State<HostState>, jar: SignedCookieJar) -> Json<Me> {
    if resolve(&s.control.config()).is_none() {
        return Json(Me {
            auth_required: false,
            authenticated: false,
            user: None,
        });
    }
    match read_session(&jar) {
        Some(session) => Json(Me {
            auth_required: true,
            authenticated: true,
            user: Some(MeUser {
                email: session.email,
                name: session.name,
            }),
        }),
        None => Json(Me {
            auth_required: true,
            authenticated: false,
            user: None,
        }),
    }
}

/// Gate `/api/*`: open when OAuth is unconfigured, else require a live session.
pub async fn require_session(
    State(s): State<HostState>,
    jar: SignedCookieJar,
    req: Request,
    next: Next,
) -> Response {
    if resolve(&s.control.config()).is_none() {
        return next.run(req).await;
    }
    if read_session(&jar).is_some() {
        next.run(req).await
    } else {
        StatusCode::UNAUTHORIZED.into_response()
    }
}

// ---- Lark calls ------------------------------------------------------------

async fn exchange_code(
    http: &reqwest::Client,
    cfg: &OAuthConfig,
    code: &str,
    verifier: &str,
    redirect_uri: &str,
) -> Result<String, String> {
    let body = serde_json::json!({
        "grant_type": "authorization_code",
        "client_id": cfg.app_id,
        "client_secret": cfg.app_secret,
        "code": code,
        "redirect_uri": redirect_uri,
        "code_verifier": verifier,
    });
    let resp = http
        .post(format!("{}/open-apis/authen/v2/oauth/token", cfg.open_base))
        .json(&body)
        .send()
        .await
        .map_err(|e| e.to_string())?;
    let status = resp.status();
    let v: serde_json::Value = resp.json().await.map_err(|e| e.to_string())?;
    info!(
        "oauth token granted scope: {:?}",
        v.get("scope").and_then(|s| s.as_str())
    );
    v.get("access_token")
        .and_then(|t| t.as_str())
        .map(str::to_string)
        .ok_or_else(|| format!("status={status} body={v}"))
}

#[derive(Deserialize, Default)]
struct UserData {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    email: Option<String>,
    #[serde(default)]
    enterprise_email: Option<String>,
    #[serde(default)]
    open_id: Option<String>,
}

impl UserData {
    /// All email-bearing fields, lowercased — matched against the allowlist.
    fn emails(&self) -> Vec<String> {
        [self.enterprise_email.as_deref(), self.email.as_deref()]
            .into_iter()
            .flatten()
            .map(|e| e.trim().to_lowercase())
            .filter(|e| !e.is_empty())
            .collect()
    }
}

#[derive(Deserialize)]
struct UserInfoResp {
    #[serde(default)]
    data: Option<UserData>,
}

async fn fetch_user(
    http: &reqwest::Client,
    cfg: &OAuthConfig,
    token: &str,
) -> Result<UserData, String> {
    let resp = http
        .get(format!("{}/open-apis/authen/v1/user_info", cfg.open_base))
        .bearer_auth(token)
        .send()
        .await
        .map_err(|e| e.to_string())?;
    let status = resp.status();
    let raw: serde_json::Value = resp.json().await.map_err(|e| e.to_string())?;
    // Field names only (plus code/msg) — values are personal data.
    info!(
        "oauth user_info code={:?} msg={:?} fields={:?}",
        raw.get("code"),
        raw.get("msg"),
        raw.get("data").and_then(|d| d.as_object()).map(|o| o
            .iter()
            .filter(|(_, v)| !v.is_null() && v.as_str() != Some(""))
            .map(|(k, _)| k.clone())
            .collect::<Vec<_>>())
    );
    let parsed: UserInfoResp = serde_json::from_value(raw).map_err(|e| e.to_string())?;
    parsed
        .data
        .ok_or_else(|| format!("no user data (status={status})"))
}

/// Load the cookie signing key: `CONSOLE_SECRET` if set, else a persisted
/// random key under the data dir (so sessions survive restarts).
pub fn load_session_key(data_dir: &Path) -> Key {
    if let Ok(secret) = std::env::var("CONSOLE_SECRET") {
        if secret.len() >= 32 {
            // Expand the secret to the 64 bytes Key::from wants.
            return Key::from(&Sha512::digest(secret.as_bytes()));
        }
        warn!("CONSOLE_SECRET is shorter than 32 bytes; ignoring it");
    }
    let path = data_dir.join("session.key");
    if let Ok(bytes) = std::fs::read(&path)
        && bytes.len() >= 64
    {
        return Key::from(&bytes);
    }
    let mut bytes = [0u8; 64];
    rand::thread_rng().fill_bytes(&mut bytes);
    match std::fs::write(&path, bytes) {
        Ok(()) => {
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                let _ = std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600));
            }
        }
        Err(e) => warn!(
            "could not persist session key to {}: {e}; sessions reset on restart",
            path.display()
        ),
    }
    Key::from(&bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pkce_challenge_matches_rfc7636() {
        // RFC 7636, Appendix B.
        let verifier = "dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk";
        assert_eq!(
            pkce_challenge(verifier),
            "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM"
        );
    }

    #[test]
    fn resolve_none_without_console_binding() {
        // a registry but no [console] lark_app => open console
        let cfg = "[lark-apps.main]\napp_id = \"cli_x\"\napp_secret = \"s\"\n";
        assert!(resolve(cfg).is_none());
    }

    #[test]
    fn resolve_none_when_bound_app_missing_or_empty() {
        assert!(resolve("[console]\nlark_app = \"ghost\"\n").is_none());
        let cfg = "[lark-apps.main]\napp_id = \"\"\napp_secret = \"\"\n\
                   [console]\nlark_app = \"main\"\n";
        assert!(resolve(cfg).is_none());
    }

    #[test]
    fn resolve_derives_hosts_and_lowercases_admins() {
        let cfg = "[lark-apps.main]\napp_id = \"cli_x\"\napp_secret = \"s\"\n\
                   [console]\nlark_app = \"main\"\n\
                   admins = [\"Alice@Example.com\", \"  \", \"BOB@x.io\"]\n";
        let o = resolve(cfg).expect("configured");
        assert_eq!(o.open_base, "https://open.larksuite.com");
        assert_eq!(o.accounts_base, "https://accounts.larksuite.com");
        assert_eq!(o.admins, vec!["alice@example.com", "bob@x.io"]);
    }

    #[test]
    fn resolve_maps_feishu_accounts_host() {
        let cfg = "[lark-apps.cn]\napp_id = \"cli_x\"\napp_secret = \"s\"\n\
                   base_url = \"https://open.feishu.cn\"\n\
                   [console]\nlark_app = \"cn\"\n";
        let o = resolve(cfg).expect("configured");
        assert_eq!(o.open_base, "https://open.feishu.cn");
        assert_eq!(o.accounts_base, "https://accounts.feishu.cn");
    }

    fn cfg_with(scope: Option<&str>, admins: &[&str]) -> OAuthConfig {
        OAuthConfig {
            app_id: "cli_x".into(),
            app_secret: "s".into(),
            open_base: "https://open.larksuite.com".into(),
            accounts_base: "https://accounts.larksuite.com".into(),
            admins: admins.iter().map(|a| a.to_string()).collect(),
            redirect_uri: None,
            scope: scope.map(str::to_string),
        }
    }

    #[test]
    fn effective_scope_defaults_to_full_set_only_with_an_allowlist() {
        // unset + allowlist => request the full identity set so emails come back
        assert_eq!(
            effective_scope(&cfg_with(None, &["a@x.io"])).as_deref(),
            Some(DEFAULT_USER_INFO_SCOPES)
        );
        // unset + open console => request nothing (bootstrap stays frictionless)
        assert_eq!(effective_scope(&cfg_with(None, &[])), None);
    }

    #[test]
    fn effective_scope_honors_explicit_override() {
        // explicit value wins verbatim (trimmed), even with an allowlist set
        assert_eq!(
            effective_scope(&cfg_with(Some("  custom:scope  "), &["a@x.io"])).as_deref(),
            Some("custom:scope")
        );
        // explicit empty opts out of any scope
        assert_eq!(effective_scope(&cfg_with(Some("   "), &["a@x.io"])), None);
    }

    #[test]
    fn user_emails_lowercased_and_filtered() {
        let u = UserData {
            name: Some("X".into()),
            email: Some("Personal@X.com".into()),
            enterprise_email: Some("  Work@Corp.com ".into()),
            open_id: None,
        };
        // enterprise email is preferred (listed first), both normalized
        assert_eq!(u.emails(), vec!["work@corp.com", "personal@x.com"]);
        assert!(UserData::default().emails().is_empty());
    }
}
