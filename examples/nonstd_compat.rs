//! Demonstrates the `nonstd-compat` feature: talking to a token endpoint
//! that doesn't quite follow RFC 6749.
//!
//! ## Case 1: dropping a field and relaxing the response `Content-Type`
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
//! ## Case 2: renaming request/response fields
//!
//! Some providers use entirely different field names on both sides of the
//! exchange, e.g. expecting `{"clientId": ..., "secret": ...}` in the
//! request body and returning `{"jwt": ..., "refreshToken": ...,
//! "expiresIn": ...}` in the response instead of this crate's
//! `access_token`/`refresh_token`/`expires_in`. Both directions are just
//! jq filters:
//! - `req_map`: `{clientId: .client_id, secret: .client_secret}` renames the
//!   credential fields (and, since the filter doesn't reference them,
//!   drops everything else this crate would otherwise have sent, such as
//!   `grant_type`).
//! - `res_map`: `{access_token: .jwt, token_type: "bearer", refresh_token:
//!   .refreshToken, expires_in: .expiresIn}` renames the response fields
//!   into the shape this crate's `StandardTokenResponse` expects,
//!   including a static `token_type` literal for a provider that never
//!   sends one.
//!
//! ## Case 3: extra params that the filter doesn't reference
//!
//! `add_extra_param()` and `add_scope()`/`add_scopes()` still work exactly
//! as they do without `nonstd-compat`: the values they add (e.g.
//! `audience`, `resource`, `scope`) become part of the same flat JSON
//! object `req_map` receives as input, under their literal key. But
//! `req_map` is a plain jq filter that produces a brand new object - any
//! input field the filter doesn't reference (explicitly or via a merge)
//! simply isn't in its output, and therefore never makes it into the
//! request body:
//! - `{clientId: .client_id, secret: .client_secret}` silently **drops**
//!   an `audience` extra param, since the filter's output object never
//!   mentions `.audience`.
//! - `. as $in | {clientId: $in.client_id, secret: $in.secret} +
//!   ($in | del(.client_id, .client_secret))` renames the credential
//!   fields *and* **forwards** everything else (`audience`, `scope`, ...)
//!   unchanged, by merging in the rest of the input object.
//!
//! ## Case 4: reshaping scopes
//!
//! `add_scope()`/`add_scopes()` populate a single space-delimited `scope`
//! field in `req_map`'s input, same as this crate would otherwise
//! form-encode. A provider that wants scopes as a JSON array instead of a
//! space-delimited string can reshape it in the filter, e.g. `{clientId:
//! .client_id, secret: .client_secret, scope: (.scope | split(" "))}`.
//!
//! ## Case 5: an invalid filter fails fast, before any network call
//!
//! If `req_map` (or `res_map`) doesn't compile as jq, the request is never
//! sent at all - preparing the request fails immediately with
//! `RequestTokenError::Other`, wrapping the jq compiler's error message.
//! This is useful to know when validating a provider config at startup,
//! rather than only discovering a typo in a filter at request time.
//!
//! ```sh
//! cargo run --example nonstd_compat --features reqwest-blocking,nonstd-compat
//! ```

use oauth2::basic::BasicClient;
use oauth2::{ClientId, ClientSecret, NonStdCompat, Scope, TokenUrl};

fn main() {
    if let Err(err) = case_1_drop_field_and_relax_content_type() {
        eprintln!("case 1 failed (expected, since example.com isn't a real endpoint): {err}");
    }
    if let Err(err) = case_2_rename_request_and_response_fields() {
        eprintln!("case 2 failed (expected, since example.com isn't a real endpoint): {err}");
    }
    if let Err(err) = case_3a_extra_param_dropped_when_filter_ignores_it() {
        eprintln!("case 3a failed (expected, since example.com isn't a real endpoint): {err}");
    }
    if let Err(err) = case_3b_extra_param_forwarded_via_merge() {
        eprintln!("case 3b failed (expected, since example.com isn't a real endpoint): {err}");
    }
    if let Err(err) = case_4_reshape_scopes_into_a_json_array() {
        eprintln!("case 4 failed (expected, since example.com isn't a real endpoint): {err}");
    }
    if let Err(err) = case_5_invalid_filter_fails_before_any_network_call() {
        eprintln!("case 5 failed as intended (invalid filter caught before sending): {err}");
    }
}

fn case_1_drop_field_and_relax_content_type() -> Result<(), Box<dyn std::error::Error>> {
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
    println!("case 1 token: {token:?}");
    Ok(())
}

fn case_2_rename_request_and_response_fields() -> Result<(), Box<dyn std::error::Error>> {
    let client = BasicClient::new(ClientId::new("aaa".to_string()))
        .set_client_secret(ClientSecret::new("bbb".to_string()))
        .set_token_uri(TokenUrl::new("https://example.com/token".to_string())?)
        .set_nonstd_compat(
            NonStdCompat::new()
                .with_req_map("{clientId: .client_id, secret: .client_secret}")
                .with_res_map(
                    "{access_token: .jwt, token_type: \"bearer\", \
                      refresh_token: .refreshToken, expires_in: .expiresIn}",
                ),
        );

    let http_client = reqwest::blocking::ClientBuilder::new()
        .redirect(reqwest::redirect::Policy::none())
        .build()?;

    let token = client.exchange_client_credentials().request(&http_client)?;
    println!("case 2 token: {token:?}");
    Ok(())
}

fn case_3a_extra_param_dropped_when_filter_ignores_it() -> Result<(), Box<dyn std::error::Error>> {
    let client = BasicClient::new(ClientId::new("aaa".to_string()))
        .set_client_secret(ClientSecret::new("bbb".to_string()))
        .set_token_uri(TokenUrl::new("https://example.com/token".to_string())?)
        .set_nonstd_compat(
            NonStdCompat::new().with_req_map("{clientId: .client_id, secret: .client_secret}"),
        );

    let http_client = reqwest::blocking::ClientBuilder::new()
        .redirect(reqwest::redirect::Policy::none())
        .build()?;

    // `audience` never appears in the request body: the filter's output
    // object only mentions `clientId`/`secret`.
    let token = client
        .exchange_client_credentials()
        .add_extra_param("audience", "https://api.example.com/resource")
        .request(&http_client)?;
    println!("case 3a token: {token:?}");
    Ok(())
}

fn case_3b_extra_param_forwarded_via_merge() -> Result<(), Box<dyn std::error::Error>> {
    let client = BasicClient::new(ClientId::new("aaa".to_string()))
        .set_client_secret(ClientSecret::new("bbb".to_string()))
        .set_token_uri(TokenUrl::new("https://example.com/token".to_string())?)
        .set_nonstd_compat(NonStdCompat::new().with_req_map(
            ". as $in | {clientId: $in.client_id, secret: $in.secret} + \
              ($in | del(.client_id, .client_secret))",
        ));

    let http_client = reqwest::blocking::ClientBuilder::new()
        .redirect(reqwest::redirect::Policy::none())
        .build()?;

    // This time `audience` (and anything else this crate would have
    // form-encoded, e.g. `grant_type`) survives, merged in unchanged
    // alongside the renamed credential fields.
    let token = client
        .exchange_client_credentials()
        .add_extra_param("audience", "https://api.example.com/resource")
        .request(&http_client)?;
    println!("case 3b token: {token:?}");
    Ok(())
}

fn case_4_reshape_scopes_into_a_json_array() -> Result<(), Box<dyn std::error::Error>> {
    let client = BasicClient::new(ClientId::new("aaa".to_string()))
        .set_client_secret(ClientSecret::new("bbb".to_string()))
        .set_token_uri(TokenUrl::new("https://example.com/token".to_string())?)
        .set_nonstd_compat(NonStdCompat::new().with_req_map(
            "{clientId: .client_id, secret: .client_secret, scope: (.scope | split(\" \"))}",
        ));

    let http_client = reqwest::blocking::ClientBuilder::new()
        .redirect(reqwest::redirect::Policy::none())
        .build()?;

    // This crate joins scopes into a single space-delimited `scope`
    // string, same as it would form-encode; the filter splits it back
    // into a JSON array of scopes for a provider that wants that shape.
    let token = client
        .exchange_client_credentials()
        .add_scope(Scope::new("read".to_string()))
        .add_scope(Scope::new("write".to_string()))
        .request(&http_client)?;
    println!("case 4 token: {token:?}");
    Ok(())
}

fn case_5_invalid_filter_fails_before_any_network_call() -> Result<(), Box<dyn std::error::Error>>
{
    let client = BasicClient::new(ClientId::new("aaa".to_string()))
        .set_client_secret(ClientSecret::new("bbb".to_string()))
        .set_token_uri(TokenUrl::new("https://example.com/token".to_string())?)
        .set_nonstd_compat(NonStdCompat::new().with_req_map("this is not valid jq"));

    let http_client = reqwest::blocking::ClientBuilder::new()
        .redirect(reqwest::redirect::Policy::none())
        .build()?;

    // Fails while preparing the request (jq fails to compile the
    // filter) - the HTTP client above is never actually called.
    let token = client.exchange_client_credentials().request(&http_client)?;
    println!("case 5 token: {token:?}");
    Ok(())
}
