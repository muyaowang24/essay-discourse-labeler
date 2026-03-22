#!/usr/bin/env bash
set -euo pipefail

essay-labeler train --config "${1:-configs/base.yaml}"
