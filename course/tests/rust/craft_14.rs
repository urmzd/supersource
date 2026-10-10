//! Why these cases exist: API version selection must be deterministic across
//! the engine and gateway while v1 and v2 clients are served side by side.
mod api_version { include!(concat!(env!("CARGO_MANIFEST_DIR"), "/../crates/tl-serve/src/api_version.rs")); }
use api_version::{cached_tokens, error_code, response_headers, select, ApiVersion, VersionError};

#[test]
// WHY: An explicit request must select its named contract over the deployment default.
// KIND: unit
fn hand_example_explicit_version_overrides_default() {
    // A request must select the contract named by its explicit version header.
    assert_eq!(select(Some("2"), ApiVersion::V1, 100, 200), Ok(ApiVersion::V2));
}
#[test]
// WHY: Unknown versions must fail closed before handler dispatch.
// KIND: boundary
fn rejects_unknown_version_before_dispatch() {
    // Unknown versions must fail closed rather than accidentally use a schema.
    assert_eq!(select(Some("3"), ApiVersion::V1, 100, 200), Err(VersionError::Unsupported));
}
#[test]
// WHY: A sunset v1 request must not reach a v2-only handler.
// KIND: boundary
fn v1_sunset_is_a_hard_boundary() {
    // Once sunset passes, v1 must not reach a handler that expects v2 fields.
    assert_eq!(select(Some("1"), ApiVersion::V2, 200, 200), Err(VersionError::Sunset));
}
#[test]
// WHY: Clients need the selected version and v1 migration deadline on responses.
// KIND: unit
fn response_version_is_echoed_and_v1_is_marked_deprecated() {
    // Clients need an observable version and migration deadline on every v1 response.
    let h=response_headers(ApiVersion::V1, 120, "Thu, 01 Jan 1970 00:03:20 GMT");
    assert!(h.contains(&("X-TL-API-Version", "1".to_string())));
    assert!(h.contains(&("Deprecation", "@120".to_string())));
    assert!(h.iter().any(|x| x.0=="Sunset"));
}
#[test]
// WHY: Versioned capacity errors must be actionable without changing v1 codes.
// KIND: unit
fn v2_error_names_are_specific_and_v1_codes_are_stable() {
    // Versioned errors preserve v1 compatibility while making v2 capacity states actionable.
    assert_eq!(error_code(ApiVersion::V1,"rate_limit_exceeded"),"rate_limit_exceeded");
    assert_eq!(error_code(ApiVersion::V2,"rate_limit_exceeded"),"requests_limit_exceeded");
    assert_eq!(error_code(ApiVersion::V2,"no_capacity"),"model_draining");
}

#[test]
// WHY: V2 requires cache usage while v1 remains schema-compatible.
// KIND: unit
fn cached_token_usage_is_present_only_in_v2() {
    // V2 makes the field mandatory; v1 remains schema-compatible and omits it.
    assert_eq!(cached_tokens(ApiVersion::V1,Some(3)),None);
    assert_eq!(cached_tokens(ApiVersion::V2,None),Some(0));
}
