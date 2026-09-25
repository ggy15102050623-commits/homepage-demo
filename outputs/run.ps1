$ErrorActionPreference = 'Stop'
$OutputDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$SessionDir = Split-Path -Parent $OutputDir
$WorkDir = Join-Path $SessionDir 'work'
$env:PYTHONIOENCODING = 'utf-8'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$env:HTTP_PROXY = 'http://127.0.0.1:7890'
$env:HTTPS_PROXY = 'http://127.0.0.1:7890'
$env:RAG_DATA_FILE = Join-Path $WorkDir 'knowledge_base_2026.jsonl'
$env:QWEN_MODEL_PATH = 'D:\AI任务\知识库\models\Qwen3-Embedding-0.6B'
$env:RAG_INDEX_FILE = Join-Path $WorkDir 'vector_index.npz'
& 'D:\Anaconda\python.exe' -X utf8 -u (Join-Path $OutputDir 'app.py')
