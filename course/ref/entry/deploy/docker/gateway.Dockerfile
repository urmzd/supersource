# The tracer gateway image (dep.00). Build from the repo root:
#   docker build -f deploy/docker/gateway.Dockerfile -t forge-gateway:0.1.0 .
FROM golang:1.24-alpine AS build
WORKDIR /src
# The module files first: `go mod download` stays cached until they change.
# go/go.mod replaces the contracts module with ../contracts/go (present once a
# unit imports it), so the whole small contracts/ tree comes along.
COPY contracts/ contracts/
COPY go/go.mod go/go.sum* go/
RUN cd go && go mod download
COPY go/ go/
RUN cd go && CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o /out/gateway ./cmd/gateway

FROM alpine:3.19
COPY --from=build /out/gateway /usr/local/bin/gateway
USER 10001:10001
# 8080: the API; 9464: /healthz and /readyz (DESIGN 2.13).
EXPOSE 8080 9464
ENTRYPOINT ["/usr/local/bin/gateway"]
