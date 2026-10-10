package gw_07

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"

	"tinyllm/gateway/auth"
	"tinyllm/gateway/ledger"
)

var admin = auth.Principal{KeyID: "adminkeyaaaa", Tenant: "ops", Scopes: []string{"admin"}}

// call serves one admin request directly on ledger.Admin, as principal p
// (nil: no authenticated principal at all).
func call(t *testing.T, h http.Handler, p *auth.Principal, method, target string) (int, http.Header, map[string]any) {
	t.Helper()
	req := httptest.NewRequest(method, target, nil)
	if p != nil {
		req = req.WithContext(auth.WithPrincipal(context.Background(), *p))
	}
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	var body map[string]any
	if rec.Body.Len() > 0 {
		if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
			t.Fatalf("%s %s: the body is not JSON: %q", method, target, rec.Body.String())
		}
	}
	return rec.Code, rec.Header(), body
}

// apiError checks the OpenAI error shape (all four keys) and returns it.
func apiError(t *testing.T, body map[string]any) map[string]any {
	t.Helper()
	e, ok := body["error"].(map[string]any)
	if !ok {
		t.Fatalf("not an error body: %v", body)
	}
	for _, k := range []string{"message", "type", "param", "code"} {
		if _, ok := e[k]; !ok {
			t.Fatalf("error object lacks %q: %v", k, e)
		}
	}
	return e
}

func TestAdminUsageEndpoint(t *testing.T) {
	// WHY: GET /admin/v1/usage is admin.v1.yaml's getUsage: the worked
	//      example's acme totals come back as {since, until, data} with one
	//      row, grouping puts only the group key in each row, and the window
	//      parameters are RFC 3339 with until exclusive. This is what
	//      `<system> usage --tenant acme --since 1h` prints.
	// KIND: conformance
	// CATCHES: s07, m02
	// CHAPTER: gw.07 section 3, Worked example by hand
	db, _ := open(t)
	recordAll(t, db, handRecords())
	h := ledger.Admin(ledger.AdminDeps{Usage: db})

	code, hdr, body := call(t, h, &admin, "GET", "/admin/v1/usage?tenant=acme")
	if code != 200 || !strings.HasPrefix(hdr.Get("Content-Type"), "application/json") {
		t.Fatalf("GET usage: %d %s", code, hdr.Get("Content-Type"))
	}
	if body["since"] != nil || body["until"] != nil {
		t.Fatalf("without a window, since and until are null: %v", body)
	}
	data := body["data"].([]any)
	row := data[0].(map[string]any)
	if len(data) != 1 || row["requests"] != 3.0 || row["errors"] != 1.0 || row["prompt_tokens"] != 32.0 ||
		row["completion_tokens"] != 40.0 || row["cached_tokens"] != 0.0 {
		t.Fatalf("acme usage: %v, want requests 3, errors 1, prompt 32, completion 40, cached 0", data)
	}
	for _, k := range []string{"tenant", "model", "key_id", "api_version"} {
		if _, ok := row[k]; ok {
			t.Fatalf("an ungrouped row has no group key, but it has %q: %v", k, row)
		}
	}

	q := url.Values{"since": {"2026-10-01T12:01:00Z"}, "until": {"2026-10-01T12:03:00Z"}, "group_by": {"model"}}
	code, _, body = call(t, h, &admin, "GET", "/admin/v1/usage?"+q.Encode())
	if code != 200 || body["since"] != "2026-10-01T12:01:00Z" || body["until"] != "2026-10-01T12:03:00Z" {
		t.Fatalf("windowed usage: %d %v", code, body)
	}
	data = body["data"].([]any)
	if len(data) != 2 {
		t.Fatalf("[12:01, 12:03) by model: %v, want rows for smol (req-2) and tiny (req-3)", data)
	}
	smol, tiny := data[0].(map[string]any), data[1].(map[string]any)
	if smol["model"] != "smol" || smol["requests"] != 1.0 || smol["prompt_tokens"] != 20.0 ||
		tiny["model"] != "tiny" || tiny["errors"] != 1.0 {
		t.Fatalf("[12:01, 12:03) by model: %v", data)
	}
	if _, ok := smol["tenant"]; ok {
		t.Fatalf("a row grouped by model carries only the model: %v", smol)
	}
}

func TestAdminUsageRejectsBadParameters(t *testing.T) {
	// WHY: a bad query parameter is a 400 whose `param` names it, so the CLI
	//      can say which flag was wrong; a silently ignored `since` would
	//      report all-time usage as the last hour's. Only GET is allowed.
	// KIND: boundary
	// CATCHES: s14, m02
	// CHAPTER: gw.07 section 2.3
	db, _ := open(t)
	h := ledger.Admin(ledger.AdminDeps{Usage: db})
	for _, c := range []struct{ query, param string }{
		{"since=yesterday", "since"},
		{"until=2026-13-01T00:00:00Z", "until"},
		{"group_by=status", "group_by"},
		{"since=2026-10-01T12:00:00Z&until=2026-10-01T12:00:00Z", "until"},
		{"since=2026-10-01T13:00:00Z&until=2026-10-01T12:00:00Z", "until"},
	} {
		code, _, body := call(t, h, &admin, "GET", "/admin/v1/usage?"+c.query)
		if code != 400 {
			t.Fatalf("?%s: status %d, want 400", c.query, code)
		}
		if e := apiError(t, body); e["param"] != c.param || e["type"] != "invalid_request_error" {
			t.Fatalf("?%s: error %v, want type invalid_request_error and param %q", c.query, e, c.param)
		}
	}
	code, hdr, _ := call(t, h, &admin, "POST", "/admin/v1/usage")
	if code != 405 || hdr.Get("Allow") != "GET" {
		t.Fatalf("POST usage: %d Allow %q, want 405 and Allow: GET", code, hdr.Get("Allow"))
	}
}

func TestAdminRequiresAdminScope(t *testing.T) {
	// WHY: usage is every tenant's billing data. The admin handler fails
	//      closed: with no authenticated principal (mounted outside the authn
	//      stage by mistake) it is a 401 invalid_api_key, and a key without
	//      the admin scope is a 403 insufficient_scope.
	// KIND: unit
	// CATCHES: s13
	// CHAPTER: gw.07 section 5, Pitfall 8
	db, _ := open(t)
	recordAll(t, db, handRecords())
	h := ledger.Admin(ledger.AdminDeps{Usage: db})
	code, _, body := call(t, h, nil, "GET", "/admin/v1/usage")
	if e := apiError(t, body); code != 401 || e["code"] != "invalid_api_key" {
		t.Fatalf("no principal: %d %v, want 401 invalid_api_key", code, e)
	}
	infer := auth.Principal{KeyID: "acmekeyaaaaa", Tenant: "acme", Scopes: []string{"infer", "embed"}}
	code, _, body = call(t, h, &infer, "GET", "/admin/v1/usage")
	if e := apiError(t, body); code != 403 || e["code"] != "insufficient_scope" || e["type"] != "permission_error" {
		t.Fatalf("an infer key: %d %v, want 403 permission_error insufficient_scope", code, e)
	}
}

func TestAdminMountsEveryPath(t *testing.T) {
	// WHY: this module owns the server side of admin.v1.yaml: each path goes to
	//      the handler of the module that implements it (keys gw.02; workers,
	//      routes, drain gw.05; cache gw.06; policy gw.08), a path whose module
	//      is not built yet is a well-formed 404 not_found, and an unknown path
	//      is a 404 too, never a fallthrough to the inference chain.
	// KIND: unit
	// CATCHES: m04
	// CHAPTER: gw.07 section 4, The interface
	served := map[string]string{}
	mark := func(name string) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			served[name] = r.Method + " " + r.URL.Path
			w.Header().Set("Content-Type", "application/json")
			io.WriteString(w, `{"ok":true}`)
		})
	}
	h := ledger.Admin(ledger.AdminDeps{Keys: mark("keys"), Workers: mark("workers"), Routes: mark("routes"),
		Drain: mark("drain"), Policy: mark("policy"), Cache: mark("cache")})
	for _, c := range []struct{ method, path, name string }{
		{"POST", "/admin/v1/keys", "keys"},
		{"DELETE", "/admin/v1/keys/acmekeyaaaaa", "keys"},
		{"GET", "/admin/v1/workers", "workers"},
		{"PUT", "/admin/v1/routes", "routes"},
		{"POST", "/admin/v1/models/smol:drain", "drain"},
		{"GET", "/admin/v1/policy", "policy"},
		{"POST", "/admin/v1/cache:purge", "cache"},
	} {
		delete(served, c.name)
		if code, _, _ := call(t, h, &admin, c.method, c.path); code != 200 || served[c.name] != c.method+" "+c.path {
			t.Fatalf("%s %s: status %d, served %q; want the %s handler", c.method, c.path, code, served[c.name], c.name)
		}
	}
	for _, p := range []string{"/admin/v1/usage", "/admin/v1/nope", "/admin/v1/models/smol"} {
		code, _, body := call(t, h, &admin, "GET", p)
		if e := apiError(t, body); code != 404 || e["code"] != "not_found" {
			t.Fatalf("GET %s: %d %v, want 404 not_found", p, code, e)
		}
	}
}
