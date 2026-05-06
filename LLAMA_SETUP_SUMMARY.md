# Llama 3.1 Model Comparison - Quick Setup Summary

## What Was Created

### 📋 Configuration Files
- **`configs/llama_models.yaml`** - Configuration for Llama 1B/3B/8B models with cluster endpoints

### 🔧 Code Modules
1. **`utils/llama_client.py`** - Cluster-based inference client for Llama models
   - `LlamaClusterClient` class for HTTP-based API calls
   - Batch inference with rate limiting
   - Async support
   - Usage statistics tracking

2. **`evaluation/compare_models.py`** - Comparison framework
   - `ModelComparison` class for running evaluations
   - Metrics: accuracy, format correctness, speed, throughput
   - Results saved as JSON
   - CLI interface for easy usage

### 📚 Documentation
- **`llama_model_comparison_notebook.ipynb`** - Interactive Jupyter notebook with:
  - Model configuration overview
  - Cluster connectivity testing
  - Full comparison execution
  - Result visualization (4 charts)
  - Quality vs speed analysis
  - Error analysis
  - Recommendations by use case

- **`LLAMA_COMPARISON_GUIDE.md`** - Comprehensive guide covering:
  - Cluster setup (vLLM, Ollama, cloud APIs)
  - Configuration
  - Usage examples
  - Result analysis
  - Troubleshooting

## Quick Start (3 Steps)

### Step 1: Set Up Cluster
```bash
# Using vLLM (recommended)
pip install vllm
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --port 8000
```

### Step 2: Update Config
Edit `configs/llama_models.yaml`:
```yaml
cluster:
  endpoint: "http://your-cluster:8000/v1"
```

### Step 3: Run Comparison
```bash
# Option A: Command line
python evaluation/compare_models.py \
    --cluster http://localhost:8000/v1 \
    --num-instances 20

# Option B: Jupyter notebook
# Open llama_model_comparison_notebook.ipynb and run cells
```

## Key Features

### 1. **LlamaClusterClient** (`utils/llama_client.py`)
```python
from utils.llama_client import get_llama_client

client = get_llama_client(
    endpoint_url="http://localhost:8000/v1",
    model_name="meta-llama/Llama-3.1-8B-Instruct"
)

# Single inference
response = client.call(messages, max_tokens=2048)

# Batch with rate limiting
responses = client.batch_call(message_batches, delay_between_calls=0.5)

# Get statistics
stats = client.get_stats()
# {'total_requests': 100, 'tokens_per_second': 45.2, ...}
```

### 2. **ModelComparison** (`evaluation/compare_models.py`)
```python
from evaluation.compare_models import ModelComparison

comparison = ModelComparison(
    config_path="configs/llama_models.yaml",
    test_file="data/processed/test.jsonl"
)

results = comparison.run_comparison(
    cluster_endpoint="http://localhost:8000/v1",
    models=["llama_1b", "llama_3b", "llama_8b"],
    num_instances=100
)

comparison.print_summary()
comparison.save_results()
```

### 3. **Interactive Notebook** (`llama_model_comparison_notebook.ipynb`)
- **Cell 1-2**: Setup and configuration
- **Cell 3**: Load model configs
- **Cell 4-5**: Test cluster connectivity
- **Cell 6-8**: Run full comparison
- **Cell 9-10**: Save and visualize results
- **Cell 11-12**: Quality vs speed analysis
- **Cell 13**: Error analysis
- **Cell 14-15**: Recommendations and export

## Expected Results (Example)

```
MODEL COMPARISON SUMMARY
================================================================================
Model       Pass Rate   Format OK   Avg Time (ms)   Tokens/sec
llama_1b    68.5%       91.2%       120.3           48.2
llama_3b    77.3%       94.8%       185.6           39.1
llama_8b    85.2%       96.9%       310.5           28.7
================================================================================

RECOMMENDATIONS BY USE CASE
✓ BEST ACCURACY: llama_8b (85.2%)
⚡ FASTEST: llama_1b (120.3ms)
⚖️  BEST BALANCE: llama_3b (Score: 0.851)
```

## File Structure

```
swerl/
├── configs/
│   └── llama_models.yaml              ← Model configurations
├── utils/
│   └── llama_client.py                ← Cluster inference client
├── evaluation/
│   └── compare_models.py              ← Comparison framework
├── llama_model_comparison_notebook.ipynb  ← Interactive notebook
├── LLAMA_COMPARISON_GUIDE.md          ← Detailed guide
└── LLAMA_SETUP_SUMMARY.md            ← This file
```

## Usage Patterns

### Pattern 1: Quick Test (5 minutes)
```python
from evaluation.compare_models import ModelComparison

comparison = ModelComparison("configs/llama_models.yaml", "data/test.jsonl")
results = comparison.run_comparison("http://localhost:8000/v1", 
                                    num_instances=10)
comparison.print_summary()
```

### Pattern 2: Full Evaluation (1-2 hours)
```bash
python evaluation/compare_models.py \
    --cluster http://localhost:8000/v1 \
    --models llama_1b llama_3b llama_8b
```

### Pattern 3: Production Deployment
```python
from utils.llama_client import get_llama_client

# Use selected model (e.g., 3B for balance)
client = get_llama_client("http://cluster:8000/v1",
                          "meta-llama/Llama-3.1-3B-Instruct")

# Integrate into your pipeline
for instance in instances:
    response = client.call(build_messages(instance), max_tokens=2048)
    # Process response...
```

## Model Selection Guide

| Model | Accuracy | Speed | Memory | Use Case |
|-------|----------|-------|--------|----------|
| **1B** | 65-70% | ⚡⚡⚡ | 2GB | Prototyping, demos, edge |
| **3B** | 75-80% | ⚡⚡ | 8GB | **Production (recommended)** |
| **8B** | 82-88% | ⚡ | 16GB | Maximum quality needed |

## Next Steps

1. ✅ **Install dependencies**: `pip install vllm` (if not using cloud API)
2. ✅ **Start cluster**: Run vLLM or Ollama server
3. ✅ **Update config**: Set `cluster.endpoint` in `llama_models.yaml`
4. ✅ **Test connection**: Run cluster connectivity test in notebook
5. ✅ **Run comparison**: Execute full model evaluation
6. ✅ **Analyze results**: Review visualizations and recommendations
7. ✅ **Deploy**: Integrate chosen model into production

## Common Issues & Solutions

**Problem**: Cluster returns 404
- **Solution**: Check endpoint URL format (should end with `/v1`)

**Problem**: Out of memory error
- **Solution**: Use smaller model (1B or 3B) or enable tensor parallelism

**Problem**: Slow inference
- **Solution**: Check GPU utilization with `nvidia-smi`; may need to adjust batch size

**Problem**: Timeouts on long instances
- **Solution**: Increase timeout: `client.timeout = 300` (5 minutes)

See `LLAMA_COMPARISON_GUIDE.md` for detailed troubleshooting.

## Code Examples by Task

### Compare Models
```python
from evaluation.compare_models import ModelComparison
comparison = ModelComparison("configs/llama_models.yaml", "data/test.jsonl")
results = comparison.run_comparison("http://localhost:8000/v1")
comparison.print_summary()
```

### Get Model Statistics
```python
from utils.llama_client import get_llama_client
client = get_llama_client("http://localhost:8000/v1", "meta-llama/Llama-3.1-8B")
stats = client.get_stats()
print(f"Tokens/sec: {stats['tokens_per_second']}")
print(f"Success rate: {stats['success_rate']}%")
```

### Batch Inference with Rate Limiting
```python
responses = client.batch_call(
    message_batches=messages_list,
    max_tokens=2048,
    delay_between_calls=1.0  # 1 second between requests
)
```

### Async Inference
```python
import asyncio
async def infer():
    response = await client.call_async(messages, max_tokens=2048)
    return response
result = asyncio.run(infer())
```

## Files to Update (Optional)

If you want to use Llama models throughout the pipeline:

1. **`run.py`**: Replace API client calls with `llama_client`
2. **`evaluation/run_inference.py`**: Use `LlamaClusterClient` instead of `OpenAIClient`
3. **`sft/generate_cot_data.py`**: Generate SFT data with Llama models

Example:
```python
# Instead of:
from utils.api_client import call_api
# Use:
from utils.llama_client import get_llama_client
client = get_llama_client("http://localhost:8000/v1", 
                          "meta-llama/Llama-3.1-8B-Instruct")
```

## Support

- **Detailed Guide**: See `LLAMA_COMPARISON_GUIDE.md`
- **Interactive Notebook**: See `llama_model_comparison_notebook.ipynb`
- **Code Documentation**: Check docstrings in `utils/llama_client.py`
- **Configuration**: Edit `configs/llama_models.yaml`

---

**Ready to compare?** Start with the Quick Start section above! 🚀
