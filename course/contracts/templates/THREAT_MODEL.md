# Threat model: <system>

Method: STRIDE over the data-flow diagram, then a ranked list of mitigations.
Scope: <the components and trust boundaries covered; the date and commit>.

## System and trust boundaries

<Embed or link docs/c4/containers.d2. Mark each trust boundary: client to
gateway, gateway to engines, worker to Python subprocesses, durable server
to its PVC, agent to tools and the web, CI to the registry.>

## Assets

| Asset | Why it matters |
|---|---|
| <API keys and the pepper> | <...> |
| <training data and the ledger> | <...> |
| <model weights> | <...> |
| <the durable WAL> | <...> |

## Threats

| Id | Component | STRIDE | Threat | Likelihood | Impact | Mitigation (implemented, file or test) | Residual risk |
|---|---|---|---|---|---|---|---|
| T1 | gateway | Spoofing | <stolen key used from elsewhere> | <L/M/H> | <L/M/H> | <...> | <...> |
| T2 | agent | Tampering | <prompt injection in a retrieved chunk triggers a write tool> | | | <ag.04 gate, injection suite> | |
| T3 | crawler | Elevation | <SSRF into the cluster network> | | | <ag.06 allowlist> | |

## Open items

- <Threats accepted or deferred, with the reason and a revisit date.>

<!-- contracts/templates/THREAT_MODEL.md (craft.17): copy to
     docs/THREAT_MODEL.md (the path craft.17 checks). -->
