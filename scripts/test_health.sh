#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-https://api.uibcdf.org}"

curl -fsS "${BASE_URL}/healthz"
