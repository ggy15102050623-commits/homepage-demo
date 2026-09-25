# 年报语义地图 RAG

基于 2026 年半年度报告构建的本地向量检索与问答演示。当前数据覆盖 9 家公司、3,576 个文本块。问答结合本地 Qwen3-Embedding-0.6B 向量、BM25 和 GPT-6 Luna，并返回原文证据块与页码。

## 仓库内容

- `outputs/index.html`：交互式语义地图与问答页面
- `outputs/app.py`：本机 Flask 服务、混合检索和 GPT-6 Luna API 调用
- `outputs/build_index.py`：使用本地 Qwen 模型重建向量索引
- `outputs/evaluation_results.csv`：10 道问题的答案、证据块和核对结果
- `outputs/页面截图.png`、`outputs/一页结论.md`：任务交付材料
- `work/knowledge_base_2026.jsonl`、`work/vector_index.npz`：年报文本和本地向量索引

## 本机运行

需使用 Windows、Anaconda Python 3.12，以及本地模型 `Qwen3-Embedding-0.6B`。默认模型路径为 `D:\AI任务\知识库\models\Qwen3-Embedding-0.6B`；如模型放在其他位置，可在启动前设置 `QWEN_MODEL_PATH`。

1. 在 `work\.secrets\openai_api_key.txt` 保存自己的 OpenAI API key。该目录已在 `.gitignore` 中排除，切勿提交密钥。
2. 安装依赖：

   ```powershell
   D:\Anaconda\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128
   D:\Anaconda\python.exe -m pip install -r .\outputs\requirements.txt
   ```

3. 确认代理监听 `127.0.0.1:7890`，运行 `outputs\run.ps1`，然后打开 `http://127.0.0.1:8765/`。

完整说明见 [`outputs/README.md`](outputs/README.md)。
