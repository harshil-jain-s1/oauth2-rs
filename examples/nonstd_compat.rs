//! Demonstrates the `nonstd-compat` feature: talking to a token endpoint
//! that doesn't quite follow RFC 6749.
//!
//! A standard `client_credentials` request from this crate would
//! form-encode a body equivalent to this JSON:
//! ```json
//! {"grant_type": "client_credentials", "client_id": "aaa", "client_secret": "bbb"}
//! ```
//! and expect an `application/json` response.
//!
//! Suppose the provider instead wants that same JSON with the `grant_type`
//! field removed (it infers the grant type from the endpoint itself), and
//! responds with `Content-Type: application/vnd.provider+json` instead of
//! `application/json`. `NonStdCompat` covers both, with no custom client
//! code required: `req_map` runs `del(.grant_type)` over the JSON this
//! crate would otherwise have form-encoded, and `res_type` relaxes the
//! response `Content-Type` check (and the `Accept` header sent).
//!
//! ```sh
//! cargo run --example nonstd_compat --features reqwest-blocking,nonstd-compat
//! ```

use oauth2::basic::BasicClient;
use oauth2::{ClientId, ClientSecret, NonStdCompat, TokenUrl};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = BasicClient::new(ClientId::new("aaa".to_string()))
        .set_client_secret(ClientSecret::new("bbb".to_string()))
        .set_token_uri(TokenUrl::new("https://example.com/token".to_string())?)
        .set_nonstd_compat(
            NonStdCompat::new()
                .with_req_map("del(.grant_type)")
                .with_res_type("application/vnd.provider+json"),
        );

    let http_client = reqwest::blocking::ClientBuilder::new()
        .redirect(reqwest::redirect::Policy::none())
        .build()?;

    let token = client.exchange_client_credentials().request(&http_client)?;
    println!("{token:?}");
    Ok(())
}
