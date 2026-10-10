#!/usr/bin/env bash
# Maintainer script: measure the reference images for dep.01's size budget
# (course/fixtures/dep.01/image-budget.json).
#
# The metric is the byte count of `docker export` of a container created from
# the image: the flattened filesystem as an uncompressed tar, which is the
# same on every image store (`docker image inspect .Size` and `docker save`
# report compressed sizes on the containerd store and uncompressed ones on
# the classic store).
#
#   course/oracle/dep.01/measure_images.sh forge-engine:ref forge-gateway:ref
set -euo pipefail
for img in "$@"; do
  c="$(docker create "$img")"
  printf '%s\t%s\n' "$img" "$(docker export "$c" | wc -c | tr -d ' ')"
  docker rm "$c" >/dev/null
done
