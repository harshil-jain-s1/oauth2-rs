//! CLI wrapper around [`oauth2::jq_login`] for exercising a config-driven
//! JSON-body login flow against a real provider. See that module's docs for
//! the full explanation of `JqLoginConfig`/`CredentialPlacement`.
//!
//! In order to run the example call:
//!
//! ```sh
//! JQ_LOGIN_URL=https://example.com/login \
//! JQ_LOGIN_CLIENT_ID=xxx \
//! JQ_LOGIN_CLIENT_SECRET=yyy \
//! JQ_LOGIN_CONFIG='{"credentials":"body","request_filter":"{client_id: .client_id, client_secret: .client_secret}","response_filter":"{access_token: .access_token}"}' \
//! cargo run --example jq_login --features reqwest-blocking,jq-login
//! ```
//!
//! If `JQ_LOGIN_CONFIG` isn't set, a built-in default config is used instead,
//! matching:
//!
//! ```sh
//! curl --location 'https://example.com/login' \
//!   --header 'Content-Type: application/json' \
//!   --data '{"client_id":"test","client_secret":"secret"}'
//! ```

use std::env;

use oauth2::jq_login::{CredentialPlacement, JqLoginClient, JqLoginConfig};
use oauth2::{ClientId, ClientSecret};
use url::Url;

fn main() -> Result<(), anyhow::Error> {
    let login_url = Url::parse(
        &env::var("JQ_LOGIN_URL").expect("Missing the JQ_LOGIN_URL environment variable."),
    )?;
    let client_id = ClientId::new(
        env::var("JQ_LOGIN_CLIENT_ID")
            .expect("Missing the JQ_LOGIN_CLIENT_ID environment variable."),
    );
    let client_secret = ClientSecret::new(
        env::var("JQ_LOGIN_CLIENT_SECRET")
            .expect("Missing the JQ_LOGIN_CLIENT_SECRET environment variable."),
    );

    // The request/response shape (credential placement, and the jq filters
    // that build the request body / normalize the response) comes from
    // JQ_LOGIN_CONFIG if set, so pointing this example at a different
    // provider is just a config string change, not a recompile.
    let config = match env::var("JQ_LOGIN_CONFIG") {
        Ok(json) => serde_json::from_str(&json)
            .map_err(|err| anyhow::anyhow!("failed to parse JQ_LOGIN_CONFIG: {err}"))?,
        Err(_) => default_config(),
    };

    let client = JqLoginClient::new(login_url, client_id, client_secret, config);

    let http_client = reqwest::blocking::ClientBuilder::new()
        // Following redirects opens the client up to SSRF vulnerabilities.
        .redirect(reqwest::redirect::Policy::none())
        .build()?;

    let token_result = client.login(&http_client)?;

    println!("{token_result:?}");

    Ok(())
}

/// The built-in config used when `JQ_LOGIN_CONFIG` isn't set.
fn default_config() -> JqLoginConfig {
    JqLoginConfig {
        credentials: CredentialPlacement::Body,
        request_filter: "{client_id: .client_id, client_secret: .client_secret}".to_string(),
        response_filter: "{access_token: .access_token, refresh_token: .refresh_token, \
                           expires_in: .expires_in}"
            .to_string(),
    }
}
