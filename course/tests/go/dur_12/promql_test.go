package dur_12

import (
	"encoding/json"
	"errors"
	"math"
	"testing"
	"time"

	"tinyllm/activities"
)

func TestHandExampleBurnRate(t *testing.T) {
	// WHY: section 3's query. Prometheus answers an instant vector with one
	//      series whose value is the string "20.5"; the activity returns
	//      20.5 and asked at the instant the workflow gave it (query and time
	//      parameters of GET /api/v1/query).
	// KIND: unit
	// CATCHES: s33
	// CHAPTER: dur.12 section 3, worked example
	p := newProm(t, 200, vector("20.5"))
	at := t0.Add(600 * time.Second)
	s, err := p.client().Query(ctx, activities.QueryInput{Expr: burnQuery, At: at})
	if err != nil || s.NoData || s.Value != 20.5 || s.Series != 1 {
		t.Fatalf("sample %+v err %v, want 20.5", s, err)
	}
	if len(p.queries) != 1 || p.queries[0]["query"] != burnQuery || p.queries[0]["time"] != "1791547800.000" {
		t.Fatalf("queries %v", p.queries)
	}
}

func TestScalarAndInfinity(t *testing.T) {
	// WHY: scalar(...) answers resultType scalar, a bare [time, "value"]
	//      pair; and +Inf (all errors over a tiny denominator) is a real,
	//      very bad burn rate, not "no data".
	// KIND: unit
	// CATCHES: s31
	// CHAPTER: dur.12 section 2, the burn check
	s, err := activities.Decode([]byte(`{"status":"success","data":{"resultType":"scalar","result":[1791547800,"3.5"]}}`))
	if err != nil || s.Value != 3.5 || s.NoData {
		t.Fatalf("scalar: %+v %v", s, err)
	}
	s, err = activities.Decode([]byte(vector("+Inf")))
	if err != nil || !math.IsInf(s.Value, 1) || s.NoData {
		t.Fatalf("+Inf: %+v %v", s, err)
	}
}

func TestNoSeriesAndNaNAreNoData(t *testing.T) {
	// WHY: an empty vector (no canary traffic matched) and NaN (0/0) both
	//      mean the check could not be computed. They must not read as 0.
	// KIND: boundary
	// CATCHES: s29
	// CHAPTER: dur.12 section 5, Pitfalls
	for _, body := range []string{emptyVector, vector("NaN")} {
		s, err := activities.Decode([]byte(body))
		if err != nil || !s.NoData {
			t.Fatalf("%s: %+v %v, want no data", body, s, err)
		}
	}
	v, nan, err := activities.ParseSample([]json.RawMessage{json.RawMessage(`1`), json.RawMessage(`"NaN"`)})
	if err != nil || !nan || v != 0 {
		t.Fatalf("ParseSample NaN: %v %v %v", v, nan, err)
	}
}

func TestSeveralSeriesAreRefused(t *testing.T) {
	// WHY: an expression that forgot sum(...) returns one series per pod;
	//      taking the first one checks a random pod. The activity refuses it,
	//      permanently: the expression must change.
	// KIND: boundary
	// CATCHES: s30, s34
	// CHAPTER: dur.12 section 5, Pitfalls
	two := `{"status":"success","data":{"resultType":"vector","result":[` +
		`{"metric":{"pod":"a"},"value":[1,"0.5"]},{"metric":{"pod":"b"},"value":[1,"30"]}]}}`
	p := newProm(t, 200, two)
	_, err := p.client().Query(ctx, activities.QueryInput{Expr: "rate(x[5m])", At: t0})
	if !errors.Is(err, activities.ErrQuery) || !nonRetryable(err) {
		t.Fatalf("two series: %v, want a non-retryable ErrQuery", err)
	}
	if _, err := activities.Decode([]byte(`{"status":"success","data":{"resultType":"matrix","result":[]}}`)); !errors.Is(err, activities.ErrQuery) {
		t.Fatalf("a range vector: %v, want ErrQuery", err)
	}
}

func TestErrorStatusIsPermanentAndServerErrorsRetry(t *testing.T) {
	// WHY: status "error" (bad_data, 400) is the expression's fault and
	//      permanent; a 503 from an overloaded Prometheus is transient and
	//      must stay retryable, or one blip rolls back a healthy canary.
	// KIND: fault
	// CATCHES: s32, s34
	// CHAPTER: dur.12 section 4, the interface
	p := newProm(t, 400, `{"status":"error","errorType":"bad_data","error":"1:5: parse error"}`)
	_, err := p.client().Query(ctx, activities.QueryInput{Expr: "sum(", At: t0})
	if !errors.Is(err, activities.ErrQuery) || !nonRetryable(err) {
		t.Fatalf("bad_data: %v, want a non-retryable ErrQuery", err)
	}
	p = newProm(t, 503, `{"status":"error","errorType":"unavailable","error":"overloaded"}`)
	_, err = p.client().Query(ctx, activities.QueryInput{Expr: burnQuery, At: t0})
	if err == nil || nonRetryable(err) {
		t.Fatalf("503: %v, want a retryable error", err)
	}
	if _, err := activities.Decode([]byte(`{"status":"error","errorType":"timeout","error":"query timed out"}`)); !errors.Is(err, activities.ErrQuery) {
		t.Fatalf("status error with 200: %v, want ErrQuery", err)
	}
}
