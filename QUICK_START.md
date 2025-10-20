# Quick Start - Gaming-Friendly Models

## ✅ Models Installed

You now have 3 models available:

1. **qwen2.5-coder:7b** (4.7 GB) ⭐ **RECOMMENDED FOR GAMING**
2. **qwen2.5-coder:3b** (1.9 GB) - Ultra-light option
3. **gpt-oss:20b** (13 GB) - Original heavyweight

## 🎮 How to Use While Gaming

### Option 1: Set Environment Variable (Temporary - Per Session)

**PowerShell:**
```powershell
$env:OLLAMA_MODEL = "qwen2.5-coder:7b"
python agent.py "Create a test package"
```

**Git Bash / WSL:**
```bash
export OLLAMA_MODEL="qwen2.5-coder:7b"
python agent.py "Create a test package"
```

### Option 2: Edit agent.py (Permanent)

Open `agent.py` and change line 19:
```python
# Before:
MODEL = os.environ.get("OLLAMA_MODEL", "gpt-oss:20b")

# After (for 7b):
MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5-coder:7b")

# Or (for ultra-fast 3b):
MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5-coder:3b")
```

## 🧪 Test the Models

Compare speeds:

```bash
# Test 7b model (recommended)
python diag_speed.py
# Result: ~7-8 seconds

# Test 3b model (ultra-fast)
OLLAMA_MODEL=qwen2.5-coder:3b python diag_speed.py
# Result: ~3-4 seconds

# Test original 20b (if you want to compare)
OLLAMA_MODEL=gpt-oss:20b python diag_speed.py
# Result: ~15-20 seconds
```

## 📊 Performance Comparison

| Model | Size | Speed | RAM Usage | Best For |
|-------|------|-------|-----------|----------|
| qwen2.5-coder:3b | 1.9 GB | 4-5x faster | ~3 GB | Heavy gaming, simple tasks |
| **qwen2.5-coder:7b** | 4.7 GB | 2-3x faster | ~6 GB | **Light gaming, most tasks** ⭐ |
| gpt-oss:20b | 13 GB | 1x (baseline) | ~16 GB | No gaming, complex tasks |

## 🚀 Example: Run Agent with New Model

```bash
# PowerShell
$env:OLLAMA_MODEL = "qwen2.5-coder:7b"
python agent.py "Create mathx package with add function and tests"

# Or Git Bash
OLLAMA_MODEL=qwen2.5-coder:7b python agent.py "Create mathx package"
```

## 💡 Recommendations

### While Gaming:
- **Light gaming** (indie games, strategy): Use `qwen2.5-coder:7b`
- **Heavy gaming** (AAA, competitive): Use `qwen2.5-coder:3b`

### Not Gaming:
- **Normal coding tasks**: Use `qwen2.5-coder:7b`
- **Complex refactoring**: Use `gpt-oss:20b`

## ⚙️ Make it Permanent

To always use the 7b model, edit `agent.py` line 19:

```python
MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5-coder:7b")
```

Then you can just run:
```bash
python agent.py "your task"
```

## 🔍 What Changed?

- **diag_speed.py**: Now defaults to `qwen2.5-coder:7b` instead of `gpt-oss:20b`
- **agent.py**: Still uses `gpt-oss:20b` by default (you can change it)
- **Both** respect the `OLLAMA_MODEL` environment variable

## Next Steps

1. **Test the model**: Run `python diag_speed.py`
2. **Try the agent**: Run a simple task with the new model
3. **Adjust if needed**: Switch to 3b if still too heavy for gaming

Enjoy coding while gaming! 🎮🚀
