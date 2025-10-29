#!/bin/bash
# This script sends a test request to the running API.

echo "Sending test request..."
curl -s -X POST http://127.0.0.1:8000/predict \
-H "Content-Type: application/json" \
-d '{"text":"This movie was fantastic, I really enjoyed it!"}'
echo ""
