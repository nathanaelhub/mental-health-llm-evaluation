#!/bin/bash
# Quick curl-based session continuation test

echo "🧪 QUICK SESSION TEST WITH CURL"
echo "================================"

BASE_URL="http://localhost:8000"
SESSION_ID="curl-test-$(date +%s)"

echo "Session ID: $SESSION_ID"
echo ""

# Test 1: First message
echo "1️⃣ First message (should trigger selection):"
echo "curl -X POST $BASE_URL/api/chat -H 'Content-Type: application/json' -d '{\"message\":\"I feel anxious\",\"session_id\":\"$SESSION_ID\",\"user_id\":\"curl-test\"}'"
echo ""

FIRST_RESPONSE=$(curl -s -X POST "$BASE_URL/api/chat" \
  -H "Content-Type: application/json" \
  -d "{\"message\":\"I feel anxious about work\",\"session_id\":\"$SESSION_ID\",\"user_id\":\"curl-test\"}")

echo "Response:"
echo "$FIRST_RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$FIRST_RESPONSE"
echo ""

# Extract selected model from first response
SELECTED_MODEL=$(echo "$FIRST_RESPONSE" | python3 -c "import sys,json; data=json.load(sys.stdin); print(data.get('selected_model','NONE'))" 2>/dev/null)
echo "Selected Model: $SELECTED_MODEL"
echo ""

# Test 2: Second message (should continue with same model)
echo "2️⃣ Second message (should continue with $SELECTED_MODEL):"
echo ""

SECOND_RESPONSE=$(curl -s -X POST "$BASE_URL/api/chat" \
  -H "Content-Type: application/json" \
  -d "{\"message\":\"Can you help me with that?\",\"session_id\":\"$SESSION_ID\",\"user_id\":\"curl-test\"}")

echo "Response:"
echo "$SECOND_RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$SECOND_RESPONSE"
echo ""

# Extract used model from second response
USED_MODEL=$(echo "$SECOND_RESPONSE" | python3 -c "import sys,json; data=json.load(sys.stdin); print(data.get('selected_model','NONE'))" 2>/dev/null)

# Compare models
echo "🔍 ANALYSIS:"
echo "First message model:  $SELECTED_MODEL"
echo "Second message model: $USED_MODEL"

if [ "$SELECTED_MODEL" = "$USED_MODEL" ] && [ "$SELECTED_MODEL" != "NONE" ]; then
    echo "✅ SUCCESS: Same model used for continuation!"
else
    echo "❌ FAILURE: Model changed or not found!"
    echo "This indicates session continuation issues."
fi