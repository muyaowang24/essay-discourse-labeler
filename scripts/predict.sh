#!/usr/bin/env bash
set -euo pipefail

essay-labeler predict --config "${1:-configs/base.yaml}" "${@:2}"
