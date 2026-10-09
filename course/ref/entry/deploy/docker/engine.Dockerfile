# The tracer engine image (dep.00). Build from the repo root:
#   docker build -f deploy/docker/engine.Dockerfile -t forge-engine:0.1.0 .
#
# Stage 1 builds libtinyllm with your c/Makefile, then tl-serve, which links
# it through tl-sys. Stage 2 ships the one binary on a small glibc base.
FROM rust:1.88-slim-bookworm AS build
RUN apt-get update \
 && apt-get install -y --no-install-recommends make \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /src
# Least to most often changed: contracts, then C, then Rust.
COPY contracts/ contracts/
COPY c/ c/
RUN make -C c
COPY rust/ rust/
RUN cargo build --release --manifest-path rust/Cargo.toml -p tl-serve \
 && cp rust/target/release/tl-serve /tl-serve

FROM debian:bookworm-slim
COPY --from=build /tl-serve /usr/local/bin/tl-serve
# Numeric, so Kubernetes runAsNonRoot can verify it.
USER 10001:10001
# 8000: POST /v1/completions; 9464: /healthz (DESIGN 2.13).
EXPOSE 8000 9464
# Exec form: tl-serve is PID 1, gets SIGTERM, and receives the chart's args.
ENTRYPOINT ["/usr/local/bin/tl-serve"]
