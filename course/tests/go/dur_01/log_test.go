// Course tests for dur.01, the append-only segmented event log
// (go/durable/log, formats/wal.md).
//
// Every test opens its own log in t.TempDir(). Durability is real (each
// acknowledged append is fsynced), but the tests pass a plain fsync(2)
// through Options.Sync: Go's (*os.File).Sync is F_FULLFSYNC on macOS, about
// 4 ms per call, which would make a 600-append test take seconds. The crash
// tests SIGKILL a child process, which loses nothing the kernel already has,
// so plain fsync is exactly as strong for what they check.
package dur_01

import (
	"bytes"
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"syscall"
	"testing"
	"time"

	dlog "tinyllm/durable/log"
)

func fastSync(f *os.File) error { return syscall.Fsync(int(f.Fd())) }

var ctx = context.Background()

func open(t *testing.T, dir string, o dlog.Options) *dlog.Log {
	t.Helper()
	if o.Sync == nil {
		o.Sync = fastSync
	}
	l, err := dlog.Open(dir, o)
	if err != nil {
		t.Fatalf("Open(%s): %v", dir, err)
	}
	t.Cleanup(func() { l.Close() })
	return l
}

func ev(typ, data string) dlog.Event { return dlog.Event{Type: typ, Data: []byte(data)} }

func mustAppend(t *testing.T, l *dlog.Log, stream string, expected int64, evs ...dlog.Event) int64 {
	t.Helper()
	v, err := l.Append(ctx, stream, expected, evs...)
	if err != nil {
		t.Fatalf("Append(%q, %d): %v", stream, expected, err)
	}
	return v
}

func readAll(t *testing.T, l *dlog.Log, stream string) []dlog.Event {
	t.Helper()
	evs, err := l.Read(ctx, stream, 1, 0)
	if err != nil {
		t.Fatalf("Read(%q): %v", stream, err)
	}
	return evs
}

func segments(t *testing.T, dir string) []string {
	t.Helper()
	m, _ := filepath.Glob(filepath.Join(dir, "*.log"))
	sort.Strings(m)
	return m
}

// independent CRC-32C (bitwise, reflected polynomial 0x82F63B78), so the
// tests never trust the code under test for a checksum.
func crc32c(b []byte) uint32 {
	c := ^uint32(0)
	for _, x := range b {
		c ^= uint32(x)
		for i := 0; i < 8; i++ {
			if c&1 == 1 {
				c = c>>1 ^ 0x82F63B78
			} else {
				c >>= 1
			}
		}
	}
	return ^c
}

func hexBytes(s string) []byte {
	var out []byte
	for _, f := range strings.Fields(s) {
		var b byte
		fmt.Sscanf(f, "%02x", &b)
		out = append(out, b)
	}
	return out
}

// The chapter's worked example: one event {stream "s", type "T", data "hi",
// at 0 ns} as the first record ever.
const handPayload = "01 00 73 01 00 00 00 00 00 00 00 01 00 00 00 01 00 54 00 00 00 00 00 00 00 00 02 00 00 00 68 69"
const handHeader = "20 00 00 00 5e 1a 72 0b 01 00 00 00 00 00 00 00"

func TestRecordHandExample(t *testing.T) {
	// WHY: the chapter's worked example byte for byte: a 32-byte batch payload
	//      and a 16-byte header whose CRC-32C (0x0B721A5E) covers the 8 seq
	//      bytes, then the payload. Recovery trusts nothing else.
	// KIND: unit, golden
	// CATCHES: s01, s02, s19
	// CHAPTER: dur.01 section 3, worked example
	payload := hexBytes(handPayload)
	got := dlog.EncodeBatch("s", 1, []dlog.Event{{Type: "T", Data: []byte("hi"), At: time.Unix(0, 0)}})
	if !bytes.Equal(got, payload) {
		t.Fatalf("EncodeBatch = % x\nwant         % x", got, payload)
	}
	var seq [8]byte
	binary.LittleEndian.PutUint64(seq[:], 1)
	if c := crc32c(append(seq[:], payload...)); c != 0x0B721A5E {
		t.Fatalf("test oracle broken: crc %08x", c)
	}
	rec := dlog.AppendRecord(nil, 1, payload)
	want := append(hexBytes(handHeader), payload...)
	if !bytes.Equal(rec, want) {
		t.Fatalf("AppendRecord = % x\nwant           % x", rec, want)
	}
	s, p, n, err := dlog.ParseRecord(append(rec, 0xFF))
	if err != nil || s != 1 || n != 48 || !bytes.Equal(p, payload) {
		t.Fatalf("ParseRecord = seq %d, n %d, err %v, payload % x", s, n, err, p)
	}
	stream, evs, err := dlog.DecodeBatch(p)
	if err != nil || stream != "s" || len(evs) != 1 || evs[0].Version != 1 || evs[0].Type != "T" ||
		string(evs[0].Data) != "hi" || evs[0].At.UnixNano() != 0 || evs[0].Stream != "s" {
		t.Fatalf("DecodeBatch = %q %+v %v", stream, evs, err)
	}
}

func TestCRC32CCheckValue(t *testing.T) {
	// WHY: CRC-32C is the Castagnoli polynomial, not the IEEE one zlib uses;
	//      its published check value over "123456789" is 0xE3069283.
	// KIND: unit
	// CATCHES: s02
	// CHAPTER: dur.01 section 2, CRC-32C
	if got := dlog.CRC32C([]byte("123456789")); got != 0xE3069283 {
		t.Fatalf("CRC32C(\"123456789\") = %08x, want e3069283", got)
	}
	if got, want := dlog.CRC32C([]byte("durable")), crc32c([]byte("durable")); got != want {
		t.Fatalf("CRC32C(\"durable\") = %08x, want %08x", got, want)
	}
}

func TestParseRecordRejectsDamage(t *testing.T) {
	// WHY: recovery decides "torn tail" or "corruption" from these errors:
	//      a short header, a length past the end, and one flipped bit
	//      anywhere in seq or payload must each be detected.
	// KIND: unit, boundary
	// CATCHES: s03
	// CHAPTER: dur.01 section 2, record format
	rec := append(hexBytes(handHeader), hexBytes(handPayload)...)
	for cut := 0; cut < len(rec); cut++ {
		if _, _, _, err := dlog.ParseRecord(rec[:cut]); !errors.Is(err, dlog.ErrShort) {
			t.Fatalf("ParseRecord(first %d bytes) = %v, want ErrShort", cut, err)
		}
	}
	for i := 8; i < len(rec); i++ { // every bit of seq and payload
		for bit := 0; bit < 8; bit++ {
			bad := append([]byte(nil), rec...)
			bad[i] ^= 1 << bit
			if _, _, _, err := dlog.ParseRecord(bad); !errors.Is(err, dlog.ErrChecksum) {
				t.Fatalf("flipping byte %d bit %d: err %v, want ErrChecksum", i, bit, err)
			}
		}
	}
}

func TestDecoderNeverPanics(t *testing.T) {
	// WHY: the decoder reads bytes a crash may have left half-written; any
	//      input must give an error or a batch that re-encodes to the same
	//      bytes, never a panic or a huge allocation. (FuzzParseRecord and
	//      FuzzDecodeBatch run the same property under `go test -fuzz`.)
	// KIND: property
	// CATCHES: s14, s19
	// CHAPTER: dur.01 section 4
	seed := hexBytes(handPayload)
	x := uint64(0x9E3779B97F4A7C15)
	next := func() uint64 { x ^= x << 13; x ^= x >> 7; x ^= x << 17; return x }
	for i := 0; i < 20000; i++ {
		b := append([]byte(nil), seed...)
		for k := int(next() % 4); k >= 0; k-- {
			switch next() % 3 {
			case 0:
				if len(b) > 0 {
					b[next()%uint64(len(b))] = byte(next())
				}
			case 1:
				b = b[:next()%uint64(len(b)+1)]
			default:
				b = append(b, byte(next()))
			}
		}
		checkDecode(t, b)
		checkParse(t, dlog.AppendRecord(nil, next()%5, b)[:next()%uint64(len(b)+17)])
	}
}

func checkDecode(t *testing.T, b []byte) {
	t.Helper()
	defer func() {
		if r := recover(); r != nil {
			t.Fatalf("DecodeBatch(% x) panicked: %v", b, r)
		}
	}()
	stream, evs, err := dlog.DecodeBatch(b)
	if err != nil {
		return
	}
	if again := dlog.EncodeBatch(stream, evs[0].Version, evs); !bytes.Equal(again, b) {
		t.Fatalf("DecodeBatch accepted % x but re-encodes to % x", b, again)
	}
}

func checkParse(t *testing.T, b []byte) {
	t.Helper()
	defer func() {
		if r := recover(); r != nil {
			t.Fatalf("ParseRecord(% x) panicked: %v", b, r)
		}
	}()
	dlog.ParseRecord(b)
}

func FuzzParseRecord(f *testing.F) {
	f.Add(append(hexBytes(handHeader), hexBytes(handPayload)...))
	f.Add([]byte{0xff, 0xff, 0xff, 0xff})
	f.Fuzz(func(t *testing.T, b []byte) {
		seq, p, n, err := dlog.ParseRecord(b)
		if err == nil && !bytes.Equal(dlog.AppendRecord(nil, seq, p), b[:n]) {
			t.Fatalf("roundtrip differs")
		}
	})
}

func FuzzDecodeBatch(f *testing.F) {
	f.Add(hexBytes(handPayload))
	f.Fuzz(func(t *testing.T, b []byte) { checkDecode(t, b) })
}

func TestAppendReadHand(t *testing.T) {
	// WHY: the stream contract: versions are 1-based and dense per stream,
	//      Append returns the new version, the log stamps At, and Read takes
	//      a start version and a limit; an unknown stream reads as empty.
	// KIND: unit
	// CATCHES: s04, s05, s06
	// CHAPTER: dur.01 section 4
	at := time.Unix(1700000000, 0)
	l := open(t, t.TempDir(), dlog.Options{Now: func() time.Time { return at }})
	if v := mustAppend(t, l, "run/a", 0, ev("Started", "x"), ev("Scheduled", "y")); v != 2 {
		t.Fatalf("first Append returned %d, want 2", v)
	}
	if v := mustAppend(t, l, "run/b", 0, ev("Started", "z")); v != 1 {
		t.Fatalf("Append to a second stream returned %d, want 1 (versions are per stream)", v)
	}
	if v := mustAppend(t, l, "run/a", 2, ev("Completed", "w")); v != 3 {
		t.Fatalf("third Append returned %d, want 3", v)
	}
	got, err := l.Read(ctx, "run/a", 2, 1)
	if err != nil || len(got) != 1 || got[0].Version != 2 || got[0].Type != "Scheduled" || string(got[0].Data) != "y" ||
		got[0].Stream != "run/a" || !got[0].At.Equal(at) {
		t.Fatalf("Read(run/a, from 2, limit 1) = %+v, %v", got, err)
	}
	all := readAll(t, l, "run/a")
	if len(all) != 3 || all[0].Version != 1 || all[2].Version != 3 || all[2].Type != "Completed" {
		t.Fatalf("Read(run/a, all) = %+v", all)
	}
	if got, _ := l.Read(ctx, "run/a", 4, 0); len(got) != 0 {
		t.Fatalf("Read past the end = %+v, want none", got)
	}
	if got, _ := l.Read(ctx, "nope", 1, 0); len(got) != 0 {
		t.Fatalf("Read of an unknown stream = %+v, want none", got)
	}
	if l.Version("run/a") != 3 || l.Version("run/b") != 1 || l.Version("nope") != 0 {
		t.Fatalf("Version = %d %d %d, want 3 1 0", l.Version("run/a"), l.Version("run/b"), l.Version("nope"))
	}
	if s := l.Streams("run/"); len(s) != 2 || s[0] != "run/a" || s[1] != "run/b" {
		t.Fatalf("Streams(run/) = %v", s)
	}
	if l.LastSeq() != 3 {
		t.Fatalf("LastSeq = %d, want 3 (one record per Append)", l.LastSeq())
	}
}

func TestAppendCopiesData(t *testing.T) {
	// WHY: the server reuses its encode buffers; an event must keep the bytes
	//      it had when Append returned, not whatever the caller writes later.
	// KIND: unit
	// CATCHES: s13
	// CHAPTER: dur.01 section 5, Pitfalls
	dir := t.TempDir()
	l := open(t, dir, dlog.Options{})
	buf := []byte("abc")
	mustAppend(t, l, "s", 0, dlog.Event{Type: "t", Data: buf})
	copy(buf, "XYZ")
	if got := readAll(t, l, "s"); string(got[0].Data) != "abc" {
		t.Fatalf("event data changed with the caller's buffer: %q", got[0].Data)
	}
}

func TestVersionConflict(t *testing.T) {
	// WHY: optimistic concurrency: two writers that both read version v cannot
	//      both append as v+1. A wrong `expected` is ErrVersionConflict and
	//      writes nothing; Any skips the check.
	// KIND: unit, boundary
	// CATCHES: s07
	// CHAPTER: dur.01 section 2, optimistic concurrency
	l := open(t, t.TempDir(), dlog.Options{})
	if _, err := l.Append(ctx, "s", 1, ev("x", "")); !errors.Is(err, dlog.ErrVersionConflict) {
		t.Fatalf("Append(expected 1) on a new stream: err %v, want ErrVersionConflict", err)
	}
	mustAppend(t, l, "s", 0, ev("x", "1"))
	for _, exp := range []int64{0, 2, 5} {
		if _, err := l.Append(ctx, "s", exp, ev("x", "bad")); !errors.Is(err, dlog.ErrVersionConflict) {
			t.Fatalf("Append(expected %d) at version 1: err %v, want ErrVersionConflict", exp, err)
		}
	}
	if l.Version("s") != 1 || l.LastSeq() != 1 {
		t.Fatalf("a rejected append wrote something: version %d, seq %d", l.Version("s"), l.LastSeq())
	}
	if v := mustAppend(t, l, "s", dlog.Any, ev("x", "2")); v != 2 {
		t.Fatalf("Append(Any) returned %d, want 2", v)
	}
}

func TestConcurrentAppendersConflict(t *testing.T) {
	// WHY: 32 goroutines race read-version-then-append on one stream. Each
	//      success must land exactly at expected+1, so every writer's events
	//      appear once and the version count equals the successes.
	// KIND: property, fault
	// CATCHES: s04, s07
	// CHAPTER: dur.01 section 4
	l := open(t, t.TempDir(), dlog.Options{})
	const writers, each = 32, 15
	var wg sync.WaitGroup
	errs := make(chan error, writers)
	for w := 0; w < writers; w++ {
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			for n := 0; n < each; {
				v := l.Version("hot")
				nv, err := l.Append(ctx, "hot", v, ev("w", fmt.Sprintf("%d/%d", w, n)))
				if errors.Is(err, dlog.ErrVersionConflict) {
					continue
				}
				if err != nil {
					errs <- err
					return
				}
				if nv != v+1 {
					errs <- fmt.Errorf("writer %d appended at expected %d but got version %d", w, v, nv)
					return
				}
				n++
			}
		}(w)
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		t.Fatal(err)
	}
	all := readAll(t, l, "hot")
	if len(all) != writers*each {
		t.Fatalf("%d events, want %d", len(all), writers*each)
	}
	seen := map[string]bool{}
	for i, e := range all {
		if e.Version != int64(i+1) || seen[string(e.Data)] {
			t.Fatalf("event %d: version %d data %q (duplicate or gap)", i, e.Version, e.Data)
		}
		seen[string(e.Data)] = true
	}
}

func TestReopenRecovers(t *testing.T) {
	// WHY: replaying every record rebuilds every stream: after Close and
	//      Open the same reads come back, and the next append continues at
	//      the next version and the next seq (no reuse after a restart).
	// KIND: unit
	// CATCHES: s08, s14
	// CHAPTER: dur.01 section 2, recovery
	dir := t.TempDir()
	l := open(t, dir, dlog.Options{})
	for i := 0; i < 5; i++ {
		mustAppend(t, l, "a", int64(2*i), ev("x", fmt.Sprint(i)), ev("y", fmt.Sprint(i)))
		mustAppend(t, l, "b", int64(i), ev("z", fmt.Sprint(i)))
	}
	before, size := readAll(t, l, "a"), l.Size()
	l.Close()
	l2 := open(t, dir, dlog.Options{})
	after := readAll(t, l2, "a")
	if len(after) != len(before) || l2.Version("b") != 5 || l2.LastSeq() != 10 || l2.Size() != size {
		t.Fatalf("after reopen: %d events in a (want %d), b at %d, seq %d, size %d (want %d)",
			len(after), len(before), l2.Version("b"), l2.LastSeq(), l2.Size(), size)
	}
	for i := range after {
		if after[i].Version != before[i].Version || after[i].Type != before[i].Type ||
			!bytes.Equal(after[i].Data, before[i].Data) || !after[i].At.Equal(before[i].At) {
			t.Fatalf("event %d differs after reopen: %+v vs %+v", i, after[i], before[i])
		}
	}
	if v := mustAppend(t, l2, "b", 5, ev("z", "5")); v != 6 || l2.LastSeq() != 11 {
		t.Fatalf("append after reopen: version %d seq %d, want 6 and 11", v, l2.LastSeq())
	}
}

func TestSegmentsRollAndNameByFirstSeq(t *testing.T) {
	// WHY: segments are named by the seq of their first record (20 digits),
	//      a new one starts before a record that would pass SegmentBytes, and
	//      recovery reads them in order as one log.
	// KIND: unit, boundary
	// CATCHES: s09, s10
	// CHAPTER: dur.01 section 3, segments
	dir := t.TempDir()
	// One record here is 16 + (3 + 8 + 4) + (3 + 8 + 4 + 10) = 56 bytes, so 3 fit in 170.
	l := open(t, dir, dlog.Options{SegmentBytes: 170})
	for i := 0; i < 7; i++ {
		mustAppend(t, l, "s", int64(i), ev("t", fmt.Sprintf("%010d", i)))
	}
	segs := segments(t, dir)
	var names []string
	for _, s := range segs {
		names = append(names, filepath.Base(s))
		st, _ := os.Stat(s)
		if st.Size() > 170 {
			t.Fatalf("%s is %d bytes, over SegmentBytes 170", filepath.Base(s), st.Size())
		}
	}
	want := []string{dlog.SegmentName(1), dlog.SegmentName(4), dlog.SegmentName(7)}
	if strings.Join(names, " ") != strings.Join(want, " ") || want[1] != "00000000000000000004.log" {
		t.Fatalf("segments %v, want %v", names, want)
	}
	l.Close()
	l2 := open(t, dir, dlog.Options{SegmentBytes: 170})
	if got := readAll(t, l2, "s"); len(got) != 7 || string(got[6].Data) != "0000000006" {
		t.Fatalf("after reopen over 3 segments: %d events", len(got))
	}
	mustAppend(t, l2, "s", 7, ev("t", "0000000007"))
	if n := len(segments(t, dir)); n != 3 {
		t.Fatalf("the 8th record should fill the third segment, found %d segments", n)
	}
}

func TestTornTailTruncatedAtEveryOffset(t *testing.T) {
	// WHY: a crash mid-append leaves a prefix of the last record. Recovery
	//      must truncate exactly there wherever the cut falls, keep every
	//      earlier record, and give the next append seq last+1.
	// KIND: property, fault
	// CATCHES: s08, s11
	// CHAPTER: dur.01 section 2, torn tail
	src := t.TempDir()
	l := open(t, src, dlog.Options{})
	mustAppend(t, l, "s", 0, ev("t", "one"))
	mustAppend(t, l, "s", 1, ev("t", "two"))
	good := l.Size()
	mustAppend(t, l, "s", 2, ev("t", "three"))
	full := l.Size()
	l.Close()
	data, _ := os.ReadFile(segments(t, src)[0])
	for cut := good; cut < full; cut++ {
		dir := t.TempDir()
		os.WriteFile(filepath.Join(dir, dlog.SegmentName(1)), data[:cut], 0o644)
		l2, err := dlog.Open(dir, dlog.Options{Sync: fastSync})
		if err != nil {
			t.Fatalf("cut at %d of %d: Open failed: %v (a torn tail is not corruption)", cut, full, err)
		}
		if l2.Version("s") != 2 || l2.LastSeq() != 2 || l2.Size() != good {
			t.Fatalf("cut at %d: version %d seq %d size %d, want 2 2 %d", cut, l2.Version("s"), l2.LastSeq(), l2.Size(), good)
		}
		if st, _ := os.Stat(filepath.Join(dir, dlog.SegmentName(1))); st.Size() != good {
			t.Fatalf("cut at %d: file is %d bytes after recovery, want %d (truncated)", cut, st.Size(), good)
		}
		if v, err := l2.Append(ctx, "s", 2, ev("t", "again")); err != nil || v != 3 || l2.LastSeq() != 3 {
			t.Fatalf("cut at %d: append after recovery = %d, %v (seq %d)", cut, v, err, l2.LastSeq())
		}
		l2.Close()
		l3, err := dlog.Open(dir, dlog.Options{Sync: fastSync})
		if err != nil || l3.Version("s") != 3 {
			t.Fatalf("cut at %d: second reopen: %v, version %d", cut, err, l3.Version("s"))
		}
		l3.Close()
	}
}

func TestGarbageTailTruncated(t *testing.T) {
	// WHY: a tail of random bytes (a write of zeros or junk that never
	//      completed) is a torn tail too; so is a whole valid record whose
	//      seq repeats the previous one, which a seq check catches and a
	//      checksum check alone does not.
	// KIND: boundary, fault
	// CATCHES: s08, s09
	// CHAPTER: dur.01 section 2, torn tail
	for name, tail := range map[string]func(last []byte) []byte{
		"zeros":     func([]byte) []byte { return make([]byte, 40) },
		"junk":      func([]byte) []byte { return []byte{0x05, 0, 0, 0, 1, 2, 3, 4, 9, 9, 9, 9, 9, 9, 9, 9, 7, 7, 7, 7, 7} },
		"repeat":    func(last []byte) []byte { return last },
		"huge len":  func([]byte) []byte { return []byte{0xff, 0xff, 0xff, 0x7f, 0, 0, 0, 0} },
		"half head": func(last []byte) []byte { return last[:9] },
	} {
		dir := t.TempDir()
		l := open(t, dir, dlog.Options{})
		mustAppend(t, l, "s", 0, ev("t", "a"))
		before := l.Size()
		mustAppend(t, l, "s", 1, ev("t", "b"))
		size := l.Size()
		l.Close()
		path := segments(t, dir)[0]
		data, _ := os.ReadFile(path)
		os.WriteFile(path, append(data, tail(data[before:])...), 0o644)
		l2, err := dlog.Open(dir, dlog.Options{Sync: fastSync})
		if err != nil {
			t.Fatalf("%s: Open: %v", name, err)
		}
		if l2.Version("s") != 2 || l2.Size() != size || l2.LastSeq() != 2 {
			t.Fatalf("%s: version %d size %d seq %d, want 2 %d 2", name, l2.Version("s"), l2.Size(), l2.LastSeq(), size)
		}
		l2.Close()
	}
}

func TestCorruptMiddleSegmentRefused(t *testing.T) {
	// WHY: damage before the last segment cannot be a crash mid-append; it
	//      is lost acknowledged data, so Open refuses with ErrCorrupt naming
	//      the file, never guessing by truncating.
	// KIND: fault, boundary
	// CATCHES: s10, s12
	// CHAPTER: dur.01 section 2, recovery
	dir := t.TempDir()
	l := open(t, dir, dlog.Options{SegmentBytes: 120})
	for i := 0; i < 6; i++ {
		mustAppend(t, l, "s", int64(i), ev("t", fmt.Sprintf("%08d", i)))
	}
	l.Close()
	segs := segments(t, dir)
	if len(segs) < 3 {
		t.Fatalf("setup: want at least 3 segments, have %d", len(segs))
	}
	data, _ := os.ReadFile(segs[1])
	data[len(data)-1] ^= 0x40
	os.WriteFile(segs[1], data, 0o644)
	_, err := dlog.Open(dir, dlog.Options{Sync: fastSync})
	if !errors.Is(err, dlog.ErrCorrupt) || !strings.Contains(err.Error(), filepath.Base(segs[1])) {
		t.Fatalf("Open over a damaged middle segment: %v, want ErrCorrupt naming %s", err, filepath.Base(segs[1]))
	}
	if st, _ := os.Stat(segs[1]); st.Size() != int64(len(data)) {
		t.Fatalf("Open modified the damaged segment (now %d bytes)", st.Size())
	}
}

func TestQuotaRejectsAndResumes(t *testing.T) {
	// WHY: at wal_max_bytes an append fails with ErrQuota and appends
	//      nothing; reads keep working, nothing acknowledged is lost, and
	//      reopening with a higher cap resumes writes (drill ops.11).
	// KIND: fault, boundary
	// CATCHES: s15, s16
	// CHAPTER: dur.01 section 2, quota
	dir := t.TempDir()
	// Each record below is 16 + (3 + 8 + 4) + (3 + 8 + 4 + 4) = 50 bytes: two fit in 100.
	l := open(t, dir, dlog.Options{MaxBytes: 100})
	mustAppend(t, l, "s", 0, ev("t", "aaaa"))
	mustAppend(t, l, "s", 1, ev("t", "bbbb"))
	if l.Size() != 100 {
		t.Fatalf("setup: size %d, want exactly the cap 100", l.Size())
	}
	_, err := l.Append(ctx, "s", 2, ev("t", "cccc"))
	if !errors.Is(err, dlog.ErrQuota) {
		t.Fatalf("append over the cap: %v, want ErrQuota", err)
	}
	if l.Size() != 100 || l.Version("s") != 2 || l.LastSeq() != 2 {
		t.Fatalf("a refused append changed the log: size %d version %d seq %d", l.Size(), l.Version("s"), l.LastSeq())
	}
	if got := readAll(t, l, "s"); len(got) != 2 {
		t.Fatalf("reads must keep working at the cap: %d events", len(got))
	}
	l.Close()
	l2 := open(t, dir, dlog.Options{MaxBytes: 1000})
	if v := mustAppend(t, l2, "s", 2, ev("t", "cccc")); v != 3 {
		t.Fatalf("after raising the cap: version %d, want 3", v)
	}
	if got := readAll(t, l2, "s"); string(got[0].Data) != "aaaa" || string(got[2].Data) != "cccc" {
		t.Fatalf("after raising the cap: %+v", got)
	}
}

func TestFailedAppendLeavesNoTrace(t *testing.T) {
	// WHY: when the write or the fsync fails, the bytes already written must
	//      be cut off. Left behind, a failed (never acknowledged) record would
	//      come back after a restart and the next good record would collide
	//      with its seq and be truncated as a torn tail: an acked append lost.
	// KIND: fault
	// CATCHES: s17
	// CHAPTER: dur.01 section 5, Pitfalls
	dir := t.TempDir()
	calls := 0
	fp := func(name string) error {
		if name != dlog.FailpointAfterWrite {
			t.Errorf("failpoint %q, want %q", name, dlog.FailpointAfterWrite)
		}
		calls++
		if calls == 2 {
			return errors.New("injected I/O error")
		}
		return nil
	}
	l := open(t, dir, dlog.Options{Failpoint: fp})
	mustAppend(t, l, "s", 0, ev("t", "one"))
	if _, err := l.Append(ctx, "s", 1, ev("t", "lost")); err == nil {
		t.Fatal("the injected failure did not fail Append")
	}
	mustAppend(t, l, "s", 1, ev("t", "two"))
	l.Close()
	l2 := open(t, dir, dlog.Options{})
	got := readAll(t, l2, "s")
	if len(got) != 2 || string(got[0].Data) != "one" || string(got[1].Data) != "two" {
		var ds []string
		for _, e := range got {
			ds = append(ds, string(e.Data))
		}
		t.Fatalf("after reopen: %v, want [one two]", ds)
	}
}

func TestSyncBeforeAck(t *testing.T) {
	// WHY: an append is acknowledged only after fsync: Sync runs once per
	//      record, after the record is in the file, before Append returns,
	//      and never for an append that was refused.
	// KIND: fault
	// CATCHES: s18
	// CHAPTER: dur.01 section 2, durability
	dir := t.TempDir()
	syncs := 0
	var lastLen int64
	l := open(t, dir, dlog.Options{Sync: func(f *os.File) error {
		syncs++
		st, err := f.Stat()
		if err != nil {
			return err
		}
		lastLen = st.Size()
		return fastSync(f)
	}})
	for i := 0; i < 3; i++ {
		mustAppend(t, l, "s", int64(i), ev("t", "x"))
		if syncs != i+1 || lastLen != l.Size() {
			t.Fatalf("after append %d: %d syncs, file %d bytes at the last sync, log size %d", i+1, syncs, lastLen, l.Size())
		}
	}
	l.Append(ctx, "s", 0, ev("t", "conflict"))
	if syncs != 3 {
		t.Fatalf("a refused append was synced (%d syncs)", syncs)
	}
}
