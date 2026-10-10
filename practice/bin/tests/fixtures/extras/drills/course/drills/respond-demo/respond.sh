#!/usr/bin/env bash
# The scripted responder (DESIGN 5.10): what a good on-call would do.
set -euo pipefail
cd "${1:?learner dir}"
kubectl --context kind-forge -n forge scale deploy/forge-engine-prefill --replicas=1 >/dev/null
mkdir -p docs/postmortems
printf '# Postmortem\n\n## Summary\n\nprefill pool scaled to zero.\n\n## Resolution\n\nscaled back to one.\n' \
  > "docs/postmortems/$(date -u +%Y-%m-%d)-respond-demo.md"
