package craft_14

import (
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
	"tinyllm/gateway/server"
)

// WHY: The selected v2 contract must reach the response and downstream meter.
// KIND: unit
func TestHandExampleExplicitV2HeaderIsEchoed(t *testing.T) {
	// The client-selected contract must be visible in the response and reach the handler.
	next := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get(server.APIVersionHeader) != "2" {
			t.Errorf("header lost")
		}
		_, _ = io.WriteString(w, "ok")
	})
	h := server.VersionedAPI(server.APIVersionConfig{Default: "1"}, next)
	r := httptest.NewRequest("GET", "/v1/models", nil)
	r.Header.Set(server.APIVersionHeader, "2")
	w := httptest.NewRecorder()
	h.ServeHTTP(w, r)
	if w.Code != 200 || w.Header().Get(server.APIVersionHeader) != "2" || w.Body.String() != "ok" {
		t.Fatalf("got %d %#v %q", w.Code, w.Header(), w.Body.String())
	}
}

// WHY: V1 clients must see deprecation and sunset headers during the overlap.
// KIND: unit
func TestV1CarriesSunsetHeadersDuringWindow(t *testing.T) {
	// V1 remains usable during the overlap, but every response announces its deadline.
	now := time.Unix(100, 0)
	h := server.VersionedAPI(server.APIVersionConfig{Default: "1", DeprecatedAt: time.Unix(90, 0), Sunset: time.Unix(200, 0), Now: func() time.Time { return now }}, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) { w.WriteHeader(204) }))
	w := httptest.NewRecorder()
	h.ServeHTTP(w, httptest.NewRequest("GET", "/v1/models", nil))
	if w.Code != 204 || w.Header().Get("Deprecation") != "@90" || w.Header().Get("Sunset") == "" {
		t.Fatalf("missing overlap metadata: %d %#v", w.Code, w.Header())
	}
}

// WHY: Sunset and unknown-version failures must keep the public error envelope.
// KIND: boundary
func TestSunsetAndUnsupportedVersionReturnOpenAIErrorShape(t *testing.T) {
	// A hard sunset and an unknown version are rejected before the downstream handler runs.
	now := time.Unix(201, 0)
	h := server.VersionedAPI(server.APIVersionConfig{Default: "2", Sunset: time.Unix(200, 0), Now: func() time.Time { return now }}, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) { t.Fatal("handler called") }))
	for _, tc := range []struct {
		version string
		status  int
		code    string
	}{{"1", 410, "api_version_sunset"}, {"7", 400, "unsupported_api_version"}} {
		r := httptest.NewRequest("GET", "/v1/models", nil)
		r.Header.Set(server.APIVersionHeader, tc.version)
		w := httptest.NewRecorder()
		h.ServeHTTP(w, r)
		if w.Code != tc.status || !strings.Contains(w.Body.String(), tc.code) || w.Header().Get("Content-Type") != "application/json" {
			t.Errorf("version %s: %d %s", tc.version, w.Code, w.Body.String())
		}
	}
}

// WHY: Per-version meter grouping must distinguish v1 traffic from v2 traffic.
// KIND: unit
func TestMeterSeesAPIVersionForGrouping(t *testing.T) {
	// A downstream meter must receive the selected version so its ledger can group usage.
	grouped := map[string]int{}
	meter := http.HandlerFunc(func(_ http.ResponseWriter, r *http.Request) { grouped[r.Header.Get(server.APIVersionHeader)]++ })
	// The absent header exercises default selection: the middleware must pass
	// the selected default downstream just as it does an explicit version.
	for _, v := range []string{"1", "2", "2", ""} {
		h := server.VersionedAPI(server.APIVersionConfig{Default: "1"}, meter)
		r := httptest.NewRequest("GET", "/v1/models", nil)
		if v != "" {
			r.Header.Set(server.APIVersionHeader, v)
		}
		h.ServeHTTP(httptest.NewRecorder(), r)
	}
	if grouped["1"] != 2 || grouped["2"] != 2 {
		t.Fatalf("meter grouped by version: %#v", grouped)
	}
}
