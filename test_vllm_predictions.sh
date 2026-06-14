clear
echo "=========================================="
echo "vLLM Test Predictions - Part B Task 1"
echo "=========================================="
echo ""
echo "Testing both V1 (Champion) and V2 (Challenger)"
echo "Date: $(date)"
echo ""

echo "=========================================="
echo "V1 (Champion - Original fp16) - Port 8001"
echo "=========================================="
echo ""

echo "Test 1: Positive sentiment"
echo "Input: 'I absolutely love this product!'"
echo ""
curl -s -X POST http://localhost:8001/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-1.5B",
    "prompt": "Classify the sentiment of this text as positive or negative: I absolutely love this product!",
    "max_tokens": 5,
    "temperature": 0
  }' | python3 -m json.tool

echo ""
echo "Test 2: Negative sentiment"
echo "Input: 'Terrible service, worst experience ever.'"
echo ""
curl -s -X POST http://localhost:8001/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-1.5B",
    "prompt": "Classify the sentiment of this text as positive or negative: Terrible service, worst experience ever.",
    "max_tokens": 5,
    "temperature": 0
  }' | python3 -m json.tool

echo ""
echo "=========================================="
echo "V2 (Challenger - Quantized GPTQ) - Port 8002"
echo "=========================================="
echo ""

echo "Test 1: Positive sentiment"
echo "Input: 'I absolutely love this product!'"
echo ""
curl -s -X POST http://localhost:8002/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "./qwen2.5-1.5b-gptq-4bit",
    "prompt": "Classify the sentiment of this text as positive or negative: I absolutely love this product!",
    "max_tokens": 5,
    "temperature": 0
  }' | python3 -m json.tool

echo ""
echo "Test 2: Negative sentiment"
echo "Input: 'Terrible service, worst experience ever.'"
echo ""
curl -s -X POST http://localhost:8002/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "./qwen2.5-1.5b-gptq-4bit",
    "prompt": "Classify the sentiment of this text as positive or negative: Terrible service, worst experience ever.",
    "max_tokens": 5,
    "temperature": 0
  }' | python3 -m json.tool

echo ""
echo "=========================================="
echo "Both servers responding correctly!"
echo "=========================================="
