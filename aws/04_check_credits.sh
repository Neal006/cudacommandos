#!/usr/bin/env bash
# Month-to-date spend. Billing API only works in us-east-1.
set -euo pipefail
aws ce get-cost-and-usage \
  --time-period Start="$(date -d "$(date +%Y-%m-01)" +%F)",End="$(date -d tomorrow +%F)" \
  --granularity MONTHLY \
  --metrics UnblendedCost \
  --region us-east-1
