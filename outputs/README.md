# 年报语义地图：本地向量 + 混合检索 RAG

## 本次数据

- 输入：原始语料 `D:\AI任务\知识库\knowledge_base_2026.jsonl`；交付包在 `work\knowledge_base_2026.jsonl` 中包含同哈希副本
- 半年度报告：9 家公司，3,576 个文本块
- 每块已有公司、章节、PDF 顺序页码和来源标题等字段；表格内容保留在原文本块中
- 嵌入：本地 `Qwen3-Embedding-0.6B`，1024 维；向量与 PCA 二维坐标保存在同级 `work\vector_index.npz`
- 检索：Qwen 语义向量 + BM25，加权 Reciprocal Rank Fusion
- 答案生成：OpenAI Responses API，模型 `gpt-6-luna`

## 启动

1. 确保模型在 `D:\AI任务\知识库\models\Qwen3-Embedding-0.6B`。交付包不复制约 1.2 GB 模型权重。
2. 在 PowerShell 执行 `outputs\run.ps1`。确保 Clash 代理监听 `127.0.0.1:7890`；服务绑定本机 `127.0.0.1:8765`，随后在浏览器打开该地址。
4. 停止服务时在服务终端按 Ctrl+C。

后端从 `work\.secrets\openai_api_key.txt` 读取 API key，并只在本机服务进程中使用。页面不会接触或返回密钥。后端仅把当前问题和召回的证据块发送到 GPT-6 Luna；完整半年报与本地向量不上传。交付包不包含密钥文件。

## 本机依赖

使用 Anaconda Python 3.12：

```powershell
D:\Anaconda\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128
D:\Anaconda\python.exe -m pip install -r .\outputs\requirements.txt
```

CPU 也能运行，但本机向量化会慢很多。

## 重建向量索引

```powershell
D:\Anaconda\python.exe -X utf8 .\outputs\build_index.py
```

可用 `--input`、`--model`、`--output` 指定数据、模型和索引路径。脚本只读取现有 JSONL，不会重下报告或改写原始知识库。输出包括向量、二维 PCA 坐标和 `work\index_manifest.json`。

## 文件

- `index.html`：交互界面
- `app.py`：本机 Flask 后端与 GPT-6 Luna API 调用
- `build_index.py`：本地 Qwen 编码和索引构建
- `eval_questions.json`、`evaluate.py`：验收题目和自动问答脚本
- `evaluation_results.csv`、`evaluation_results.json`：逐题答案、召回块、页码、引用及正确性判定
- `run.ps1`：启动本机服务
- `一页结论.md`：问答测试结论
- `页面截图.png`：本机页面截图

API 模型 ID：`gpt-6-luna`。官方文档：[GPT-6 Luna API](https://developers.openai.com/api/docs/models/gpt-6-luna)
