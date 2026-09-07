#!/usr/bin/env bash
# Download a pinned CI tool: URL, expected SHA-256, destination.
# Never install a partial download or execute an unverified response body.
set -euo pipefail

if [[ "$#" != 3 ]]; then
    echo "Usage: $0 URL SHA256 DESTINATION" >&2
    exit 2
fi

download_tmp=$(mktemp "${3}.download.XXXXXX")
trap 'rm -f -- "$download_tmp"' EXIT

curl --fail --show-error --location \
    --retry 3 --retry-all-errors --retry-delay 1 --retry-max-time 120 \
    --connect-timeout 10 --max-time 30 \
    --output "$download_tmp" "$1"
printf '%s  %s\n' "$2" "$download_tmp" | shasum -a 256 --check --strict -
mv -- "$download_tmp" "$3"
