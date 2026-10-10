// Course tests for go/config: defaults, then runtime.toml, then the
// TL_<SECTION>__<KEY> environment (contracts/config/runtime.schema.json).
// Every case passes its own env slice, so the process environment of the
// test run never leaks in.
package gw_01

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"tinyllm/config"
	"tinyllm/gateway/server"
)

const handFile = `
[gateway]
listen = ":8081"
cache_entries = 100
route_policy = "least_outstanding"

[[gateway.routes]]
model = "tinystories-10m"
targets = ["unified"]

[[gateway.routes]]
model = "smol-135m"
cascade = [{ model = "tinystories-10m", accept_if = "mean_logprob > -1.2" }, { model = "smol-135m" }]

[engine]
speculative = { draft = "ngram", k = 4 }

[ext]
my_flag = "anything goes here"
`

func TestConfigHandExample(t *testing.T) {
	// WHY: the chapter's precedence example: the default (4096 entries) is
	//      replaced by the file (100), which is replaced by the environment
	//      (50); keys nobody set keep their defaults.
	// KIND: unit
	// CATCHES: s13
	// CHAPTER: gw.01 section 3, worked example (configuration)
	env := []string{"TL_GATEWAY__CACHE_ENTRIES=50", "TL_API_KEY=tl_x_y", "HOME=/home/me"}
	c, err := config.Parse([]byte(handFile), env)
	if err != nil {
		t.Fatal(err)
	}
	if c.Gateway.CacheEntries != 50 {
		t.Fatalf("cache_entries = %d, want 50 (env beats file beats default)", c.Gateway.CacheEntries)
	}
	if c.Gateway.Listen != ":8081" || c.Gateway.RoutePolicy != "least_outstanding" {
		t.Fatalf("file values lost: listen %q, route_policy %q", c.Gateway.Listen, c.Gateway.RoutePolicy)
	}
	if c.Gateway.HealthListen != ":9464" || c.Gateway.KeyCacheTTLS != 5 || c.Gateway.AffinityLoadFactor != 1.25 || c.Engine.KVBlocks != 2048 {
		t.Fatalf("defaults lost: %+v", c.Gateway)
	}
	if len(c.Gateway.Routes) != 2 || len(c.Gateway.Routes[1].Cascade) != 2 ||
		c.Gateway.Routes[1].Cascade[0].AcceptIf != "mean_logprob > -1.2" || c.Gateway.Routes[0].Targets[0] != "unified" {
		t.Fatalf("routes %+v", c.Gateway.Routes)
	}
	if c.Engine.Speculative.Draft != "ngram" || c.Ext["my_flag"] != "anything goes here" {
		t.Fatalf("inline table or [ext] lost: %+v %v", c.Engine.Speculative, c.Ext)
	}
}

func TestConfigDefaults(t *testing.T) {
	// WHY: an empty file is a valid runtime.toml: every schema default must
	//      already be in place (the milestone runner's templates rely on it).
	// KIND: unit
	// CHAPTER: gw.01 section 2.5
	c, err := config.Parse(nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	checks := map[string]bool{
		"paths.artifacts":         c.Paths.Artifacts == "/artifacts",
		"gateway.listen":          c.Gateway.Listen == ":8080",
		"gateway.registry_listen": c.Gateway.RegistryListen == ":50060",
		"gateway.pepper_env":      c.Gateway.PepperEnv == "TL_GATEWAY_PEPPER",
		"gateway.drain_deadline":  c.Gateway.DrainDeadlineS == 30,
		"gateway.heartbeat_miss":  c.Gateway.HeartbeatMissLimit == 3,
		"engine.role":             c.Engine.Role == "unified",
		"durable.wal_max_bytes":   c.Durable.WALMaxBytes == 2147483648,
		"otel.trace_sample_ratio": c.OTel.TraceSampleRatio == 1.0,
	}
	for k, ok := range checks {
		if !ok {
			t.Errorf("default %s is wrong", k)
		}
	}
}

func TestConfigSeparator(t *testing.T) {
	// WHY: keys contain single underscores (kv_blocks), so only the double
	//      underscore separates section from key; TL_API_KEY and
	//      TL_GATEWAY_PEPPER are secrets, not overrides, and must be ignored.
	// KIND: unit
	// CATCHES: s12
	// CHAPTER: gw.01 section 2.5
	env := []string{
		"TL_ENGINE__KV_BLOCKS=4096",
		"TL_GATEWAY_LISTEN=:1",     // single underscore: not an override
		"TL_GATEWAY_PEPPER=secret", // a secret, not an override
		"tl_engine__max_seqs=1",    // not TL_
		"TL_DURABLE__WAL_MAX_BYTES=1024",
		"TL_OTEL__ENDPOINT=http://collector:4318/?a=b", // '=' inside the value
	}
	c, err := config.Parse(nil, env)
	if err != nil {
		t.Fatal(err)
	}
	if c.Engine.KVBlocks != 4096 || c.Durable.WALMaxBytes != 1024 {
		t.Fatalf("kv_blocks %d, wal_max_bytes %d", c.Engine.KVBlocks, c.Durable.WALMaxBytes)
	}
	if c.Gateway.Listen != ":8080" || c.Engine.MaxSeqs != 64 {
		t.Fatalf("non-overrides were applied: listen %q, max_seqs %d", c.Gateway.Listen, c.Engine.MaxSeqs)
	}
	if c.OTel.Endpoint != "http://collector:4318/?a=b" {
		t.Fatalf("endpoint %q: split NAME=value at the first '=' only", c.OTel.Endpoint)
	}
	if got := config.EnvName("engine", "kv_blocks"); got != "TL_ENGINE__KV_BLOCKS" {
		t.Fatalf("EnvName = %q", got)
	}
}

func TestConfigEnvTypes(t *testing.T) {
	// WHY: an override is parsed as its key's type: an integer key rejects
	//      "lots", a number key accepts 1.5.
	// KIND: unit
	// CATCHES: s12
	// CHAPTER: gw.01 section 2.5
	c, err := config.Parse(nil, []string{"TL_GATEWAY__AFFINITY_LOAD_FACTOR=1.5", "TL_ENGINE__THREADS=0"})
	if err != nil || c.Gateway.AffinityLoadFactor != 1.5 {
		t.Fatalf("affinity_load_factor %v, %v", c.Gateway.AffinityLoadFactor, err)
	}
	_, err = config.Parse(nil, []string{"TL_ENGINE__KV_BLOCKS=lots"})
	if err == nil || !strings.Contains(err.Error(), "TL_ENGINE__KV_BLOCKS") {
		t.Fatalf("a non-integer kv_blocks must fail naming the variable, got %v", err)
	}
}

func TestConfigEnvErrors(t *testing.T) {
	// WHY: a typo in an override must stop the service at start-up, not be
	//      silently ignored; tables and arrays are not overridable.
	// KIND: boundary
	// CATCHES: s15, s16
	// CHAPTER: gw.01 section 5, Pitfalls, item 8
	for _, v := range []string{
		"TL_GATEWAY__NOPE=1",           // unknown key
		"TL_NOPE__X=1",                 // unknown section
		"TL_ENGINE__SPECULATIVE=ngram", // a table
		"TL_GATEWAY__ROUTES=[]",        // an array
	} {
		_, err := config.Parse(nil, []string{v})
		name, _, _ := strings.Cut(v, "=")
		if err == nil || !strings.Contains(err.Error(), name) {
			t.Fatalf("%s: err = %v, want an error naming the variable", v, err)
		}
	}
}

func TestConfigRejectsUnknownKeys(t *testing.T) {
	// WHY: runtime.schema.json forbids unknown keys outside [ext]; a typo such
	//      as cache_entires would otherwise leave the default in force.
	// KIND: boundary
	// CATCHES: s14
	// CHAPTER: gw.01 section 5, Pitfalls, item 7
	for _, doc := range []string{"[gateway]\ncache_entires = 5\n", "[gatway]\nlisten = \":1\"\n", "[engine.speculative]\ndraft = \"none\"\nkk = 1\n"} {
		if _, err := config.Parse([]byte(doc), nil); err == nil {
			t.Fatalf("%q: an unknown key must be an error", doc)
		}
	}
	if _, err := config.Parse([]byte("[ext]\nanything = 1\n[ext.deeper]\nx = [1, 2]\n"), nil); err != nil {
		t.Fatalf("[ext] is the learner's own: %v", err)
	}
}

func TestConfigValidation(t *testing.T) {
	// WHY: the enumerations hold after the environment too: an override can
	//      set route_policy = "random" as easily as the file can.
	// KIND: boundary
	// CATCHES: s17
	// CHAPTER: gw.01 section 2.5
	if _, err := config.Parse([]byte("[gateway]\nroute_policy = \"random\"\n"), nil); err == nil {
		t.Fatal("route_policy = random must be rejected")
	}
	if _, err := config.Parse(nil, []string{"TL_GATEWAY__ROUTE_POLICY=random"}); err == nil {
		t.Fatal("TL_GATEWAY__ROUTE_POLICY=random must be rejected")
	}
	if _, err := config.Parse([]byte("[gateway]\nkey_cache_ttl_s = 9\n"), nil); err == nil {
		t.Fatal("key_cache_ttl_s = 9 is outside 0 to 5")
	}
}

func TestConfigLoadAndFromRuntime(t *testing.T) {
	// WHY: Load is the entry point your main calls; FromRuntime turns
	//      drain_deadline_s into the server's drain deadline.
	// KIND: unit
	// CATCHES: s13
	// CHAPTER: gw.01 section 4
	p := filepath.Join(t.TempDir(), "runtime.toml")
	if err := os.WriteFile(p, []byte("[gateway]\nlisten = \":9\"\ndrain_deadline_s = 7\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	c, err := config.Load(p, []string{"TL_GATEWAY__HEALTH_LISTEN=:10"})
	if err != nil {
		t.Fatal(err)
	}
	sc := server.FromRuntime(c.Gateway)
	if sc.Listen != ":9" || sc.HealthListen != ":10" || sc.DrainDeadline != 7*time.Second {
		t.Fatalf("FromRuntime = %+v", sc)
	}
	if _, err := config.Load(filepath.Join(t.TempDir(), "missing.toml"), nil); err == nil {
		t.Fatal("a missing file is an error")
	}
}
