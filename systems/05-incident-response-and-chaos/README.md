# Incident Response and Chaos

## Overview

- **Primary reference**: [*Site Reliability Engineering*](https://sre.google/sre-book/table-of-contents/) (Google, free): ch. 12 Effective Troubleshooting, ch. 14 Managing Incidents, ch. 15 Postmortem Culture
- **Supplementary**: [*The Site Reliability Workbook*](https://sre.google/workbook/table-of-contents/) (free), ch. 9 Incident Response and ch. 10 Postmortem Culture; *Chaos Engineering* by Rosenthal and Jones (O'Reilly); Kubernetes docs, [Debug Applications](https://kubernetes.io/docs/tasks/debug/debug-application/) (free)
- **Prerequisites**: [Cloud Native](../03-cloud-native/), [Observability](../04-observability/)
- **Estimated time**: one drill per pass of the course, 1 to 2 h each; the full set in Pass 11

## Key Takeaways

- Incidents are learned by causing them on purpose: a drill injects a known fault into your own running system and grades how you detect, mitigate, and record it.
- Mitigate first (roll back, fail over, shed load), find the root cause second, write it down third.
- A runbook is written before the incident, for the person who did not build the system; a postmortem is written after, without blame.
- Detection moves from a person noticing (Pass 1) to burn-rate alerts (Pass 7), and `ss drill end` measures time to detect and time to mitigate from Prometheus.

## How to Study

- Do each drill when its pass reaches it, against your own kind cluster, with the time limit on.
- Write the runbook during the drill, not after: the commands you actually typed are the Diagnosis section.
- Read one chapter of the SRE book after each drill and compare its advice with what you did.

---

# Concepts & Techniques

## Core Insight

Every system fails; what differs is how long it stays failed. Time to mitigate is mostly time spent working out what is wrong, so the work that pays is making the next diagnosis shorter: telemetry that shows the cause, runbooks that say where to look, and rollouts that can be undone in one command.

## 1. Triage on Kubernetes

**Key ideas**:
- **Outside in**: which workload is unhealthy, why its last container exited (exit code and reason), what its previous log says, what changed (rollout history).
- **Exit codes**: below 128 the program chose to exit; `128 + n` means signal `n` (137 SIGKILL, usually OOMKilled; 143 SIGTERM).
- **CrashLoopBackOff**: restarts wait 10 s, doubling to 5 minutes.

## 2. Mitigation

**Key ideas**:
- **Roll back** the change that started it (`kubectl rollout undo`, `helm rollback`), then fix forward through the chart.
- **Readiness and rollout strategy** decide whether a bad rollout becomes an outage.

## 3. Learning from incidents

**Key ideas**:
- **Blameless postmortems**: Summary, Impact, Timeline, Root cause, Detection, Resolution, Action items.
- **Action items** that prevent, detect, or mitigate the next occurrence, each with an owner.

## Chapters

<!-- ss:chapters -->
| # | Module | Chapter | Kind | Pass |
|---|---|---|---|---|
| 1 | `ops.00` | [First drill: the engine crashloops](00-first-drill.md) | drill | 1 |
<!-- /ss:chapters -->

## Connections to Other Tracks

| Track | Connection |
|---|---|
| [Observability](../04-observability/) | the SLOs and burn-rate alerts that detect what these drills inject |
| [Cloud Native](../03-cloud-native/) | the Kubernetes objects every drill acts on |
| [Software Craftsmanship: documentation](../../software-craftsmanship/06-documentation-writing/) | runbooks and decision records |

## Company Relevance

| Company | Practice |
|---|---|
| Google | SRE incident command and blameless postmortems |
| Netflix | chaos engineering against production |
| Any on-call team | runbooks linked from alerts; game days |
