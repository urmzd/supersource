// Log correlation (obs.01, used by obs.02): every log line written inside a
// span carries that span's trace_id and span_id, so `kubectl logs | grep
// <trace id>` finds a request's lines (semconv.md, Logs).

package otelx

import (
	"context"
	"log/slog"
)

// LogHandler wraps next so that a record logged with a context holding a
// valid span (slog.InfoContext(ctx, ...), Logger.Log(ctx, ...)) gains the
// attributes trace_id and span_id, lowercase hex. Records without one pass
// through unchanged. With slog.NewJSONHandler as next, a line reads
// {"time":...,"level":"INFO","msg":"...","trace_id":"4bf9...","span_id":"00f0..."}.
func LogHandler(next slog.Handler) slog.Handler {
	// SOLUTION-BEGIN obs.01
	return logHandler{next}
	// SOLUTION-END
}

type logHandler struct{ next slog.Handler }

func (h logHandler) Enabled(ctx context.Context, l slog.Level) bool {
	// SOLUTION-BEGIN obs.01
	return h.next.Enabled(ctx, l)
	// SOLUTION-END
}

func (h logHandler) Handle(ctx context.Context, r slog.Record) error {
	// SOLUTION-BEGIN obs.01
	if tid, sid, ok := IDs(ctx); ok {
		r = r.Clone()
		r.AddAttrs(slog.String("trace_id", tid), slog.String("span_id", sid))
	}
	return h.next.Handle(ctx, r)
	// SOLUTION-END
}

// WithAttrs and WithGroup keep the wrapper around the derived handler, or
// logger.With(...) would silently drop the ids.
func (h logHandler) WithAttrs(as []slog.Attr) slog.Handler {
	// SOLUTION-BEGIN obs.01
	return logHandler{h.next.WithAttrs(as)}
	// SOLUTION-END
}

func (h logHandler) WithGroup(name string) slog.Handler {
	// SOLUTION-BEGIN obs.01
	return logHandler{h.next.WithGroup(name)}
	// SOLUTION-END
}
