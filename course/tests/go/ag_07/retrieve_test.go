// Course tests for ag.07: retrieval over a RAG index (contracts/formats/
// rag-index.md). BM25 and the bm25.idx bytes are compared with an
// independent Python oracle (course/oracle/ag.07/make_fixtures.py), the
// fusion and diversity rules with numbers worked by hand in the chapter, and
// IVF with exact search on synthetic clustered vectors.
//
// No network, no model: queries are embedded by the fixture's own
// feature-hashing embedder (helpers_test.go).
package ag_07

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"tinyllm/agent/rag/retrieve"
	"tinyllm/ds/rng"
)

var handDocs = []string{"kv cache", "kv cache eviction policy", "gateway rate limits"}

func TestHandExampleBM25(t *testing.T) {
	// WHY: the chapter's worked example (section 3): N = 3, avgdl = 3, query
	//      "cache eviction". idf(cache) = ln 1.6, idf(eviction) = ln(8/3);
	//      the short document gets more per "cache" than the long one, but
	//      the long one wins by also holding "eviction"; the third document
	//      matches nothing and is not returned.
	// KIND: unit
	// CATCHES: s01
	// CHAPTER: ag.07 section 3, worked example
	x := retrieve.BuildBM25(handDocs, 1.2, 0.75)
	if x.AvgDL != 3 {
		t.Fatalf("avgdl = %v, want 3", x.AvgDL)
	}
	if got := x.IDF("cache"); !near(got, 0.47000362924573563, 1e-12) {
		t.Fatalf("idf(cache) = %.17g, want ln 1.6 = 0.47000362924573563", got)
	}
	if got := x.IDF("eviction"); !near(got, 0.9808292530117263, 1e-12) {
		t.Fatalf("idf(eviction) = %.17g, want ln(8/3) = 0.9808292530117263", got)
	}
	s := x.Scores("cache eviction")
	want := []float64{0.5442147286003255, 1.2767329363865667, 0}
	for i := range want {
		if !near(s[i], want[i], 1e-12) {
			t.Fatalf("scores = %v, want %v", s, want)
		}
	}
	got := x.Search("cache eviction", 10, nil)
	if len(got) != 2 || got[0].Doc != 1 || got[1].Doc != 0 {
		t.Fatalf("Search = %+v, want doc 1 then doc 0 (doc 2 matches nothing)", got)
	}
}

func TestTerms(t *testing.T) {
	// WHY: the term rule is the contract between ingest and retrieval: if
	//      the index was built with one tokenization and queries use
	//      another, "Café" never finds "café". Runs of Unicode letters,
	//      digits, and underscore, lowercased; everything else separates.
	// KIND: boundary
	// CATCHES: s04, s05
	// CHAPTER: ag.07 section 2.1
	cases := []struct {
		in   string
		want []string
	}{
		{"KV cache, eviction!", []string{"kv", "cache", "eviction"}},
		{"tl_kv_export(handle)", []string{"tl_kv_export", "handle"}},
		{"k-means++ at 10x", []string{"k", "means", "at", "10x"}},
		{"Café ÜBERALL naïve", []string{"café", "überall", "naïve"}},
		{"東京 2026", []string{"東京", "2026"}},
		{"   ", nil},
		{"", nil},
	}
	for _, c := range cases {
		got := retrieve.Terms(c.in)
		if len(got) == 0 && len(c.want) == 0 {
			continue
		}
		if !reflect.DeepEqual(got, c.want) {
			t.Errorf("Terms(%q) = %q, want %q", c.in, got, c.want)
		}
	}
}

func TestBM25MatchesOracle(t *testing.T) {
	// WHY: the contract's acceptance test: every chunk's score for eleven
	//      queries equals an independent implementation to 1e-9, including
	//      a repeated query term ("cache cache" doubles), accented terms, an
	//      identifier with underscores, and a query matching nothing.
	// KIND: conformance
	// CATCHES: s01, s02, s03, s04, s09
	// CHAPTER: ag.07 section 4
	e := loadExpected(t)
	x := retrieve.BuildBM25(fixtureTexts(t), e.K1, e.B)
	if !near(x.AvgDL, e.AvgDL, 1e-12) || len(x.Postings) != e.NTerms {
		t.Fatalf("avgdl %v, %d terms; want %v, %d", x.AvgDL, len(x.Postings), e.AvgDL, e.NTerms)
	}
	for _, q := range e.Queries {
		s := x.Scores(q.Q)
		for i := range q.BM25 {
			if !near(s[i], q.BM25[i], 1e-9) {
				t.Fatalf("%q: chunk %d scores %.17g, want %.17g", q.Q, i, s[i], q.BM25[i])
			}
		}
		got := x.Search(q.Q, 10, nil)
		if len(got) != len(q.LexicalTop) {
			t.Fatalf("%q: %d hits, want %d", q.Q, len(got), len(q.LexicalTop))
		}
		for i, w := range q.LexicalTop {
			if got[i].Doc != int(w[0]) {
				t.Fatalf("%q: hit %d is chunk %d, want %d", q.Q, i, got[i].Doc, int(w[0]))
			}
		}
	}
}

func TestBM25IdxGoldenBytes(t *testing.T) {
	// WHY: bm25.idx is a file format other code reads (your ingest writes
	//      it, a restarted retriever reads it). The bytes for the fixture
	//      corpus must equal the oracle's: header, doc lengths, terms sorted
	//      by bytes, gap-encoded LEB128 postings.
	// KIND: golden
	// CATCHES: s06, s07
	// CHAPTER: ag.07 section 2.2
	e := loadExpected(t)
	x := retrieve.BuildBM25(fixtureTexts(t), e.K1, e.B)
	var buf bytes.Buffer
	if _, err := x.WriteTo(&buf); err != nil {
		t.Fatal(err)
	}
	want, err := os.ReadFile(filepath.Join(fixtureDir(t), "index", "bm25.idx"))
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(buf.Bytes(), want) {
		n := 0
		for n < len(want) && n < buf.Len() && want[n] == buf.Bytes()[n] {
			n++
		}
		t.Fatalf("bm25.idx differs from the oracle at byte %d (len %d, want %d)", n, buf.Len(), len(want))
	}
}

func TestBM25IdxRoundTrip(t *testing.T) {
	// WHY: reading the file back must give the index that was written:
	//      same postings, same doc lengths, same avgdl bits, so a restarted
	//      retriever ranks exactly as before.
	// KIND: property
	// CATCHES: s06, s07
	// CHAPTER: ag.07 section 2.2
	e := loadExpected(t)
	for _, texts := range [][]string{handDocs, fixtureTexts(t)} {
		x := retrieve.BuildBM25(texts, e.K1, e.B)
		var buf bytes.Buffer
		x.WriteTo(&buf)
		y, err := retrieve.ReadBM25(&buf, e.K1, e.B)
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(x.DocLen, y.DocLen) || x.AvgDL != y.AvgDL || !reflect.DeepEqual(x.Postings, y.Postings) {
			t.Fatalf("round trip changed the index")
		}
	}
}

func TestUvarint(t *testing.T) {
	// WHY: postings are LEB128 varints: seven bits per byte, low group
	//      first, high bit = more bytes follow. The boundary values 127/128
	//      and 16383/16384 are where a wrong shift or mask shows; a value cut
	//      off mid-way must be reported (n = 0), not read as a small number.
	// KIND: boundary
	// CATCHES: s06
	// CHAPTER: ag.07 section 2.2
	cases := []struct {
		v    uint64
		want []byte
	}{
		{0, []byte{0x00}},
		{1, []byte{0x01}},
		{127, []byte{0x7f}},
		{128, []byte{0x80, 0x01}},
		{300, []byte{0xac, 0x02}},
		{16383, []byte{0xff, 0x7f}},
		{16384, []byte{0x80, 0x80, 0x01}},
		{1<<32 - 1, []byte{0xff, 0xff, 0xff, 0xff, 0x0f}},
		{1<<64 - 1, []byte{0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x01}},
	}
	for _, c := range cases {
		got := retrieve.PutUvarint(nil, c.v)
		if !bytes.Equal(got, c.want) {
			t.Errorf("PutUvarint(%d) = % x, want % x", c.v, got, c.want)
		}
		v, n := retrieve.Uvarint(append(got, 0x55))
		if v != c.v || n != len(c.want) {
			t.Errorf("Uvarint(% x) = %d, %d; want %d, %d", got, v, n, c.v, len(c.want))
		}
		if len(c.want) > 1 {
			if _, n := retrieve.Uvarint(c.want[:len(c.want)-1]); n != 0 {
				t.Errorf("Uvarint of truncated % x returned n = %d, want 0", c.want, n)
			}
		}
	}
}

func TestReadBM25RejectsCorrupt(t *testing.T) {
	// WHY: an index file is input from disk: a crash mid-write leaves a
	//      prefix of it. Every truncation, a wrong magic, and unsorted terms
	//      must be an error wrapping ErrFormat, never a panic and never a
	//      silently smaller index.
	// KIND: fault
	// CATCHES: s08
	// CHAPTER: ag.07 section 5
	good, err := os.ReadFile(filepath.Join(fixtureDir(t), "index", "bm25.idx"))
	if err != nil {
		t.Fatal(err)
	}
	try := func(name string, b []byte) {
		defer func() {
			if r := recover(); r != nil {
				t.Fatalf("%s: ReadBM25 panicked: %v", name, r)
			}
		}()
		if _, err := retrieve.ReadBM25(bytes.NewReader(b), 1.2, 0.75); !errors.Is(err, retrieve.ErrFormat) {
			t.Fatalf("%s: err = %v, want ErrFormat", name, err)
		}
	}
	for n := 0; n < len(good); n += 7 {
		try(fmt.Sprintf("truncated to %d bytes", n), good[:n])
	}
	try("truncated by one byte", good[:len(good)-1])
	bad := append([]byte(nil), good...)
	copy(bad, "TLBX")
	try("bad magic", bad)
	try("trailing bytes", append(append([]byte(nil), good...), 0))
	x := retrieve.BuildBM25([]string{"bb aa"}, 1.2, 0.75)
	var buf bytes.Buffer
	x.WriteTo(&buf)
	swapped := buf.Bytes()
	// the two term entries are "aa" then "bb" (each 2+2+4+4+2 bytes): swap them
	off := 24 + 4
	ent := 2 + 2 + 4 + 4 + 2
	a := append([]byte(nil), swapped[off:off+ent]...)
	copy(swapped[off:], swapped[off+ent:off+2*ent])
	copy(swapped[off+ent:], a)
	try("unsorted terms", swapped)
}

func TestOpenIndex(t *testing.T) {
	// WHY: Open is where the four files must agree: docs.jsonl line i is
	//      row i of vectors.f32 and document i of bm25.idx. A missing row,
	//      a short vector file, or a duplicate chunk id silently pairs the
	//      wrong text with a score, so each is an ErrIndex.
	// KIND: unit
	// CATCHES: s19
	// CHAPTER: ag.07 section 4
	dir := filepath.Join(fixtureDir(t), "index")
	x, err := retrieve.Open(dir)
	if err != nil {
		t.Fatal(err)
	}
	if len(x.Chunks) != 39 || len(x.Vecs) != 39 || len(x.Vecs[0]) != 16 || x.BM25.N() != 39 {
		t.Fatalf("opened %d chunks, %d vectors of dim %d, %d bm25 docs; want 39, 39, 16, 39", len(x.Chunks), len(x.Vecs), len(x.Vecs[0]), x.BM25.N())
	}
	if c, ok := x.Chunk("web/ivf.html#1"); !ok || c.DocID != "web/ivf.html" || c.Source != "web" {
		t.Fatalf("Chunk(web/ivf.html#1) = %+v, %v", c, ok)
	}
	if _, ok := x.Chunk("web/ivf.html#9"); ok {
		t.Fatal("Chunk found an id the index does not hold")
	}
	broken := func(name string, edit func(d string)) {
		d := t.TempDir()
		for _, f := range []string{"meta.json", "docs.jsonl", "vectors.f32", "bm25.idx"} {
			b, _ := os.ReadFile(filepath.Join(dir, f))
			os.WriteFile(filepath.Join(d, f), b, 0o644)
		}
		edit(d)
		if _, err := retrieve.Open(d); !errors.Is(err, retrieve.ErrIndex) && !errors.Is(err, retrieve.ErrFormat) {
			t.Fatalf("%s: Open err = %v, want ErrIndex", name, err)
		}
	}
	broken("short vectors", func(d string) {
		p := filepath.Join(d, "vectors.f32")
		b, _ := os.ReadFile(p)
		os.WriteFile(p, b[:len(b)-4], 0o644)
	})
	broken("extra vectors", func(d string) {
		p := filepath.Join(d, "vectors.f32")
		b, _ := os.ReadFile(p)
		os.WriteFile(p, append(b, make([]byte, 64)...), 0o644)
	})
	broken("missing docs line", func(d string) {
		p := filepath.Join(d, "docs.jsonl")
		b, _ := os.ReadFile(p)
		lines := bytes.Split(bytes.TrimSpace(b), []byte("\n"))
		os.WriteFile(p, bytes.Join(lines[:len(lines)-1], []byte("\n")), 0o644)
	})
	broken("duplicate chunk id", func(d string) {
		p := filepath.Join(d, "docs.jsonl")
		b, _ := os.ReadFile(p)
		lines := bytes.Split(bytes.TrimSpace(b), []byte("\n"))
		lines[1] = lines[0]
		os.WriteFile(p, bytes.Join(lines, []byte("\n")), 0o644)
	})
}

func TestFlatMatchesOracle(t *testing.T) {
	// WHY: exact search is the yardstick IVF is measured against, so it
	//      must itself be exact: the k largest dot products with the
	//      normalized query, ties to the lower chunk.
	// KIND: conformance
	// CATCHES: s09
	// CHAPTER: ag.07 section 2.3
	e := loadExpected(t)
	x, err := retrieve.Open(filepath.Join(fixtureDir(t), "index"))
	if err != nil {
		t.Fatal(err)
	}
	emb := &hashEmbedder{dim: e.Dim}
	for _, q := range e.Queries {
		raw, _ := emb.Embed(context.Background(), []string{q.Q})
		got := (&retrieve.Flat{Vecs: x.Vecs}).Search(retrieve.Normalize(raw[0]), 10, nil)
		for i, w := range q.FlatTop {
			if got[i].Doc != int(w[0]) || !near(got[i].Score, w[1], 1e-12) {
				t.Fatalf("%q: flat hit %d = %+v, want chunk %d score %.17g", q.Q, i, got[i], int(w[0]), w[1])
			}
		}
	}
}

func TestKMeansPPMatchesOracle(t *testing.T) {
	// WHY: k-means++ seeding is specified down to the draw: the first seed
	//      is Below(n), each next one is the first row whose running sum of
	//      squared distances exceeds Float64() * total. With the same PCG32
	//      stream the picks equal the oracle's, so IVF centroids are
	//      reproducible across machines and languages.
	// KIND: conformance
	// CATCHES: s10, s11
	// CHAPTER: ag.07 section 2.4
	e := loadExpected(t)
	x, err := retrieve.Open(filepath.Join(fixtureDir(t), "index"))
	if err != nil {
		t.Fatal(err)
	}
	for _, c := range e.KMeansPP {
		got, err := retrieve.KMeansPP(x.Vecs, c.K, rng.Seeded(c.Seed))
		if err != nil || !reflect.DeepEqual(got, c.Picks) {
			t.Fatalf("KMeansPP(k=%d, seed=%d) = %v, %v; want %v", c.K, c.Seed, got, err, c.Picks)
		}
	}
	if _, err := retrieve.KMeansPP(x.Vecs, 40, rng.Seeded(1)); !errors.Is(err, retrieve.ErrK) {
		t.Fatalf("k > rows: err = %v, want ErrK", err)
	}
}

func TestIVFSeededByIndexID(t *testing.T) {
	// WHY: IVF centroids are not stored (rag-index.md): every load builds
	//      them again, seeded with fnv1a64(index_id). With zero Lloyd
	//      rounds the centroids are exactly the k-means++ seed rows of that
	//      stream, and two loads build identical lists.
	// KIND: unit
	// CATCHES: s13
	// CHAPTER: ag.07 section 2.4
	e := loadExpected(t)
	dir := filepath.Join(fixtureDir(t), "index")
	if got := retrieve.FNV1a64("docs"); got != e.KMeansPP[0].Seed {
		t.Fatalf("FNV1a64(docs) = %d, want %d", got, e.KMeansPP[0].Seed)
	}
	for _, s := range []struct {
		in   string
		want uint64
	}{{"", 0xcbf29ce484222325}, {"a", 0xaf63dc4c8601ec8c}, {"foobar", 0x85944171f73967e8}} {
		if got := retrieve.FNV1a64(s.in); got != s.want {
			t.Fatalf("FNV1a64(%q) = %#x, want %#x", s.in, got, s.want)
		}
	}
	x, _ := retrieve.Open(dir)
	if err := x.BuildIVF(4, 0); err != nil {
		t.Fatal(err)
	}
	for c, row := range e.KMeansPP[0].Picks {
		if !reflect.DeepEqual(x.IVF.Centroids[c], x.Vecs[row]) {
			t.Fatalf("centroid %d is not row %d (the k-means++ pick for seed fnv1a64(\"docs\"))", c, row)
		}
	}
	y, _ := retrieve.Open(dir)
	x.BuildIVF(4, 10)
	y.BuildIVF(4, 10)
	if !reflect.DeepEqual(x.IVF.Lists, y.IVF.Lists) || !reflect.DeepEqual(x.IVF.Centroids, y.IVF.Centroids) {
		t.Fatal("two loads of the same index built different IVF lists")
	}
}

func TestIVFRecall(t *testing.T) {
	// WHY: the catalog's acceptance bar: on clustered data (2000 unit
	//      vectors, 32 clusters, dim 32) IVF with 32 lists and nprobe = 8
	//      finds at least 95% of the exact top 10, and probing every list is
	//      exact. A probe that scans the wrong lists, or only one, falls
	//      far below.
	// KIND: property
	// CATCHES: s12
	// CHAPTER: ag.07 section 2.4
	vecs := clustered(11, 2000, 32, 32, 1.2)
	queries := clustered(11, 2100, 32, 32, 1.2)[2000:] // same centres, new points
	ivf, err := retrieve.BuildIVF(vecs, 32, 10, rng.Seeded(42))
	if err != nil {
		t.Fatal(err)
	}
	n := 0
	for _, l := range ivf.Lists {
		n += len(l)
	}
	if n != len(vecs) {
		t.Fatalf("lists hold %d rows, want %d", n, len(vecs))
	}
	flat := &retrieve.Flat{Vecs: vecs}
	var r8, rAll float64
	for _, q := range queries {
		exact := flat.Search(q, 10, nil)
		r8 += retrieve.Recall(ivf.Search(q, 10, 8, nil), exact)
		rAll += retrieve.Recall(ivf.Search(q, 10, 32, nil), exact)
	}
	r8 /= float64(len(queries))
	rAll /= float64(len(queries))
	t.Logf("recall@10: nprobe=8 %.3f, nprobe=32 %.3f", r8, rAll)
	if r8 < 0.95 {
		t.Fatalf("recall@10 at nprobe=8 = %.3f, want >= 0.95", r8)
	}
	if rAll != 1 {
		t.Fatalf("recall@10 probing all 32 lists = %.3f, want 1", rAll)
	}
}

func h(i int) retrieve.Hit { return retrieve.Hit{Index: i, ChunkID: fmt.Sprintf("d#%d", i)} }

func TestRRFHandExample(t *testing.T) {
	// WHY: the chapter's fusion example: lexical [x=0, y=1, z=2], dense
	//      [z=2, x=0], k = 60. x scores 1/61 + 1/62, z 1/63 + 1/61, y 1/62:
	//      order x, z, y. Ranks count from 1, so a hit missing from a list
	//      adds nothing for it, and equal fused scores go to the lower index.
	// KIND: unit
	// CATCHES: s14, s15
	// CHAPTER: ag.07 section 3
	got := retrieve.RRF([][]retrieve.Hit{{h(0), h(1), h(2)}, {h(2), h(0)}}, 60, 0)
	want := []struct {
		idx   int
		score float64
	}{{0, 1.0/61 + 1.0/62}, {2, 1.0/61 + 1.0/63}, {1, 1.0 / 62}}
	if len(got) != 3 {
		t.Fatalf("RRF = %+v", got)
	}
	for i, w := range want {
		if got[i].Index != w.idx || !near(got[i].Score, w.score, 1e-15) || got[i].Rank != i+1 {
			t.Fatalf("RRF[%d] = %+v, want index %d score %.17g rank %d", i, got[i], w.idx, w.score, i+1)
		}
	}
	tie := retrieve.RRF([][]retrieve.Hit{{h(7), h(3)}, {h(3), h(7)}}, 60, 1)
	if len(tie) != 1 || tie[0].Index != 3 {
		t.Fatalf("tie RRF cut to 1 = %+v, want index 3 (equal scores go to the lower index)", tie)
	}
}

func TestMMRHandExample(t *testing.T) {
	// WHY: the chapter's diversity example: q = (0.8, 0.6), candidates
	//      c0 = (1, 0), c1 = (0.96, 0.28), c2 = (0, 1), lambda = 0.5.
	//      Relevance order is c1, c0, c2, but after c1 is picked, c0 is
	//      almost a copy of it (sim 0.96), so MMR takes c2 second:
	//      values 0.468, 0.16, -0.08.
	// KIND: unit
	// CATCHES: s16
	// CHAPTER: ag.07 section 3
	vecs := map[int][]float32{0: {1, 0}, 1: {0.96, 0.28}, 2: {0, 1}}
	got := retrieve.MMR([]float32{0.8, 0.6}, []retrieve.Hit{h(0), h(1), h(2)}, func(i int) []float32 { return vecs[i] }, 0.5, 3)
	want := []struct {
		idx int
		v   float64
	}{{1, 0.468}, {2, 0.16}, {0, -0.08}}
	for i, w := range want {
		if got[i].Index != w.idx || !near(got[i].Score, w.v, 1e-6) || got[i].Rank != i+1 {
			t.Fatalf("MMR[%d] = %+v, want index %d value %v", i, got[i], w.idx, w.v)
		}
	}
	if top := retrieve.MMR([]float32{0.8, 0.6}, []retrieve.Hit{h(0), h(1), h(2)}, func(i int) []float32 { return vecs[i] }, 1, 3); top[1].Index != 0 {
		t.Fatalf("lambda = 1 is pure relevance: second pick %d, want 0", top[1].Index)
	}
}

func TestHybridMatchesOracle(t *testing.T) {
	// WHY: the whole retriever against the oracle: BM25 and flat lists of
	//      50 candidates, RRF with k = 60, cut to 5; the same with a
	//      sources filter; and MMR (lambda 0.5, k 4) over the fused list,
	//      which only matches if the query vector is normalized first.
	// KIND: conformance
	// CATCHES: s14, s17
	// CHAPTER: ag.07 section 4
	e := loadExpected(t)
	x, err := retrieve.Open(filepath.Join(fixtureDir(t), "index"))
	if err != nil {
		t.Fatal(err)
	}
	check := func(name string, got []retrieve.Hit, want [][2]any) {
		t.Helper()
		if len(got) != len(want) {
			t.Fatalf("%s: %d hits, want %d", name, len(got), len(want))
		}
		for i, w := range want {
			if got[i].ChunkID != w[0].(string) || !near(got[i].Score, w[1].(float64), 1e-12) || got[i].Rank != i+1 {
				t.Fatalf("%s: hit %d = %s %.17g rank %d, want %s %.17g", name, i, got[i].ChunkID, got[i].Score, got[i].Rank, w[0], w[1])
			}
		}
	}
	ctx := context.Background()
	for _, q := range e.Queries {
		hy := &retrieve.Hybrid{Index: x, Embedder: &hashEmbedder{dim: e.Dim}}
		got, err := hy.Retrieve(ctx, q.Q, 5, retrieve.Filter{})
		if err != nil {
			t.Fatal(err)
		}
		check(q.Q, got, q.Hybrid5)
		got, _ = hy.Retrieve(ctx, q.Q, 5, retrieve.Filter{Sources: []string{"web"}})
		check(q.Q+" (web only)", got, q.Hybrid5Web)
		hy.Lambda = 0.5
		got, _ = hy.Retrieve(ctx, q.Q, 4, retrieve.Filter{})
		check(q.Q+" (mmr)", got, q.MMR4)
	}
}

func TestFilterBeforeCut(t *testing.T) {
	// WHY: a filter applied after the top-n cut returns fewer than k hits
	//      (or none) even though k chunks pass it: with 3 candidates per
	//      list and a filter on one document, every hit must still come
	//      from that document and there must be 3 of them.
	// KIND: boundary
	// CATCHES: s18
	// CHAPTER: ag.07 section 5
	x, err := retrieve.Open(filepath.Join(fixtureDir(t), "index"))
	if err != nil {
		t.Fatal(err)
	}
	hy := &retrieve.Hybrid{Index: x, Embedder: &hashEmbedder{dim: 16}, Candidates: 3}
	got, err := hy.Retrieve(context.Background(), "kv cache eviction", 3, retrieve.Filter{DocIDs: []string{"web/ivf.html"}})
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 3 {
		t.Fatalf("%d hits, want 3 (web/ivf.html has 3 chunks)", len(got))
	}
	for _, g := range got {
		if g.DocID != "web/ivf.html" {
			t.Fatalf("hit %s is outside the filter", g.ChunkID)
		}
	}
	if !(retrieve.Filter{}).Keep(x.Chunks[0]) || (retrieve.Filter{Sources: []string{"web"}}).Keep(x.Chunks[0]) {
		t.Fatal("Filter.Keep: empty filter must keep everything; a sources filter must drop other sources")
	}
}

type badEmbedder struct{ dim, n int }

func (b badEmbedder) Embed(_ context.Context, texts []string) ([][]float32, error) {
	out := make([][]float32, b.n)
	for i := range out {
		out[i] = make([]float32, b.dim)
		out[i][0] = 1
	}
	return out, nil
}

func TestEmbedderMismatch(t *testing.T) {
	// WHY: the query must be embedded by the model that embedded the index
	//      (meta.json embedding_model). A query vector of another dimension
	//      is an error that names both dims, not a panic and not a
	//      truncated dot product; so is an embedder that returns no vector.
	// KIND: boundary
	// CATCHES: s20
	// CHAPTER: ag.07 section 5
	x, err := retrieve.Open(filepath.Join(fixtureDir(t), "index"))
	if err != nil {
		t.Fatal(err)
	}
	for _, b := range []badEmbedder{{dim: 8, n: 1}, {dim: 32, n: 1}, {dim: 16, n: 0}} {
		func() {
			defer func() {
				if r := recover(); r != nil {
					t.Fatalf("embedder %+v: Retrieve panicked: %v", b, r)
				}
			}()
			hy := &retrieve.Hybrid{Index: x, Embedder: b}
			if _, err := hy.Retrieve(context.Background(), "cache", 3, retrieve.Filter{}); err == nil {
				t.Fatalf("embedder %+v: no error", b)
			}
		}()
	}
}

func TestLexicalOnlyAndEmpty(t *testing.T) {
	// WHY: without an embedder the retriever is BM25 alone through the same
	//      fusion; a query that matches nothing returns no hits and no
	//      error, and k = 0 asks for nothing (the agent's tool must handle
	//      "no results" as an answer, not a failure).
	// KIND: smoke
	// CHAPTER: ag.07 section 4
	x, err := retrieve.Open(filepath.Join(fixtureDir(t), "index"))
	if err != nil {
		t.Fatal(err)
	}
	hy := &retrieve.Hybrid{Index: x}
	got, err := hy.Retrieve(context.Background(), "port 30080", 3, retrieve.Filter{})
	if err != nil || len(got) != 1 || got[0].ChunkID != "docs/numbers.md#0" || !near(got[0].Score, 1.0/61, 1e-15) {
		t.Fatalf("lexical-only = %+v, %v; want docs/numbers.md#0 with 1/61", got, err)
	}
	if got, err := hy.Retrieve(context.Background(), "zebra quantum", 3, retrieve.Filter{}); err != nil || len(got) != 0 {
		t.Fatalf("no match = %+v, %v; want no hits, no error", got, err)
	}
	if got, err := hy.Retrieve(context.Background(), "cache", 0, retrieve.Filter{}); err != nil || len(got) != 0 {
		t.Fatalf("k = 0: %+v, %v", got, err)
	}
}
