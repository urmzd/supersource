package gw_07

import (
	"bufio"
	"context"
	"database/sql"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	"tinyllm/gateway/ledger"
)

// childEnv tells the test binary, re-executed as a child process, to act as
// a gateway that records forever until it is killed.
const childEnv = "GW07_LEDGER_CHILD"

// killRecord is the deterministic row number i: the parent recomputes it to
// check that every row on disk is complete.
func killRecord(i int) ledger.UsageRecord {
	ttft := time.Duration(i%97+1) * time.Millisecond
	return ledger.UsageRecord{
		RequestID: fmt.Sprintf("kill-%06d", i), Start: t0.Add(time.Duration(i) * time.Millisecond),
		Tenant: []string{"acme", "globex", "initech"}[i%3], KeyID: "acmekeyaaaaa", Model: "smol", ServedModel: "smol",
		Route: "/v1/chat/completions", Status: 200, Stream: i%2 == 0,
		PromptTokens: i%50 + 1, CompletionTokens: (i * 7) % 40, CachedTokens: i % 5,
		TTFT: &ttft, E2E: time.Duration(i%200+100) * time.Millisecond, WorkerID: "w-" + strconv.Itoa(i%4),
	}
}

func TestHelperLedgerWriter(t *testing.T) {
	// WHY: not a check on its own: the child process of
	//      TestSIGKILLLeavesRowsCompleteOrAbsent. It records rows forever and
	//      prints "acked <i>" only after Record returned, so the parent knows
	//      which rows the gateway believed were safe when it was killed.
	// KIND: fault
	// CHAPTER: gw.07 section 4, What the tests check
	path := os.Getenv(childEnv)
	if path == "" {
		t.Skip("helper process for TestSIGKILLLeavesRowsCompleteOrAbsent")
	}
	db, err := ledger.Open(path)
	if err != nil {
		fmt.Println("open failed:", err)
		os.Exit(3)
	}
	w := bufio.NewWriter(os.Stdout)
	for i := 0; ; i++ {
		if err := db.Record(context.Background(), killRecord(i)); err != nil {
			fmt.Fprintln(w, "record failed:", err)
			w.Flush()
			os.Exit(4)
		}
		fmt.Fprintln(w, "acked", i)
		w.Flush()
	}
}

func TestSIGKILLLeavesRowsCompleteOrAbsent(t *testing.T) {
	// WHY: a gateway pod is SIGKILLed on OOM or a failed liveness probe, with
	//      no chance to flush anything. Every row Record acknowledged must be
	//      on disk and complete (all 19 columns as written), the file must pass
	//      integrity_check, and it must still be in WAL mode: a ledger that
	//      batches rows in memory or splits a row over two statements loses or
	//      tears exactly the rows the kill interrupts.
	// KIND: fault
	// CATCHES: s05, s16
	// CHAPTER: gw.07 section 2.4
	path := filepath.Join(t.TempDir(), "usage.db")
	cmd := exec.Command(os.Args[0], "-test.run=^TestHelperLedgerWriter$", "-test.count=1")
	cmd.Env = append(os.Environ(), childEnv+"="+path)
	out, err := cmd.StdoutPipe()
	if err != nil {
		t.Fatal(err)
	}
	cmd.Stderr = os.Stderr
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	acked := -1
	sc := bufio.NewScanner(out)
	deadline := time.Now().Add(15 * time.Second)
	for sc.Scan() {
		line := sc.Text()
		if !strings.HasPrefix(line, "acked ") {
			if strings.Contains(line, "failed") {
				_ = cmd.Process.Kill()
				_ = cmd.Wait()
				t.Fatalf("the writer failed before the kill: %s", line)
			}
			continue
		}
		acked, _ = strconv.Atoi(strings.TrimPrefix(line, "acked "))
		if acked >= 400 || time.Now().After(deadline) {
			break
		}
	}
	if err := cmd.Process.Kill(); err != nil { // SIGKILL on Unix
		t.Fatal(err)
	}
	for sc.Scan() { // acks already in the pipe were printed before the kill
		if n, err := strconv.Atoi(strings.TrimPrefix(sc.Text(), "acked ")); err == nil && n > acked {
			acked = n
		}
	}
	_ = cmd.Wait()
	if acked < 50 {
		t.Fatalf("only %d rows acknowledged in 15 s: Record is far too slow", acked+1)
	}

	db := raw(t, path)
	if ic := pragma(t, db, "integrity_check"); ic != "ok" {
		t.Fatalf("integrity_check after SIGKILL: %s", ic)
	}
	if jm := pragma(t, db, "journal_mode"); jm != "wal" {
		t.Fatalf("journal_mode after SIGKILL = %q, want wal", jm)
	}
	rows, err := db.Query(`SELECT request_id, ts_ms, tenant, key_id, model, served_model, route, api_version, status,
		error_code, stream, prompt_tokens, completion_tokens, cached_tokens, ttft_ms, e2e_ms, cache_hit, worker_id, trace_id FROM usage`)
	if err != nil {
		t.Fatal(err)
	}
	defer rows.Close()
	seen := map[int]bool{}
	for rows.Next() {
		var (
			id, tenant, key, model, served, route, ver, worker, trace string
			ts                                                        int64
			status, stream, p, c, ca, hit                             int
			code                                                      sql.NullString
			ttft                                                      sql.NullFloat64
			e2e                                                       float64
		)
		if err := rows.Scan(&id, &ts, &tenant, &key, &model, &served, &route, &ver, &status, &code, &stream,
			&p, &c, &ca, &ttft, &e2e, &hit, &worker, &trace); err != nil {
			t.Fatal(err)
		}
		i, err := strconv.Atoi(strings.TrimPrefix(id, "kill-"))
		if err != nil {
			t.Fatalf("unexpected row %q", id)
		}
		w := killRecord(i)
		b2i := map[bool]int{false: 0, true: 1}
		if ts != w.Start.UnixMilli() || tenant != w.Tenant || key != w.KeyID || model != w.Model || served != w.ServedModel ||
			route != w.Route || ver != "1" || status != w.Status || code.Valid || stream != b2i[w.Stream] ||
			p != w.PromptTokens || c != w.CompletionTokens || ca != w.CachedTokens || !ttft.Valid ||
			ttft.Float64 != float64(*w.TTFT)/float64(time.Millisecond) || e2e != float64(w.E2E)/float64(time.Millisecond) ||
			hit != 0 || worker != w.WorkerID || trace != "" {
			t.Fatalf("row %s is incomplete or wrong after SIGKILL: tokens %d/%d/%d ttft %v e2e %v worker %q",
				id, p, c, ca, ttft, e2e, worker)
		}
		seen[i] = true
	}
	for i := 0; i <= acked; i++ {
		if !seen[i] {
			t.Fatalf("row kill-%06d was acknowledged before the SIGKILL but is not on disk (%d of %d acked rows present)",
				i, len(seen), acked+1)
		}
	}
}
