# 本機 RAG 問答助手

這是一個使用 Streamlit、Ollama 與 FAISS 製作的本機 RAG 學習專案。使用者可上傳一份或多份 PDF／PNG／JPG；系統會保留原生文字與 OCR，並可選擇使用 VLM 建立獨立的視覺描述 Chunk，再依檢索內容回答問題並標示來源檔名、頁碼與內容類型。

> 本專案僅供學習與資訊整理，不構成法律或合規建議。請勿上傳機密、個人資料或其他不應交由本機模型處理的內容。

## 使用技術

- Python：主要開發語言。
- Streamlit：建立本機 Web 操作介面，提供 PDF 上傳、知識庫建立、文件問答與 Chunk 預覽。
- LangChain：整合 RAG 流程中的文件載入、文字切分、Embedding、向量庫與聊天模型呼叫。
- PyPDFLoader：解析 PDF，將 PDF 頁面轉成 LangChain `Document`。
- PyMuPDF：將需要 OCR 的 PDF 頁面轉成圖片。
- PaddleOCR：辨識掃描型 PDF 頁面與 PNG/JPG 圖片中的中文、英文及數字。
- RecursiveCharacterTextSplitter：將長文本依段落、換行與中文標點切成適合檢索的 Chunks。
- Ollama：在本機執行 Embedding 模型與聊天模型。
- bge-m3：Embedding 模型，負責將文字 Chunk 轉成語意向量。
- FlagEmbedding：載入 Hugging Face Reranker 並計算 query/document 相關性分數。
- BAAI/bge-reranker-v2-m3：第二階段重排模型，將 FAISS 候選 Chunk 重新評分與排序。
- qwen3:4b：聊天模型，負責根據檢索到的文件內容產生繁體中文回答。
- qwen3-vl:4b：預設 VLM，在建庫階段理解圖片、表格、圖表與可見異常，輸出結構化繁體中文描述。
- Pydantic：驗證 VLM 的 JSON Schema 輸出。
- FAISS：向量資料庫，用於儲存文字向量並執行相似度搜尋。
- python-dotenv：讀取 `.env` 設定檔。
- pytest：撰寫與執行單元測試。

## RAG 流程

```text
PDF / PNG / JPG 上傳
  ├─ PyPDFLoader／PaddleOCR → Text Chunks
  └─ 頁面圖片／上傳圖片 → qwen3-vl:4b → Visual Chunks
  ↓
RecursiveCharacterTextSplitter 分別切分 Text／Visual Chunk
  ↓
bge-m3 產生 Embedding
  ↓
FAISS 建立向量索引
  ↓
使用者提問
  ↓
FAISS 取回相關 Chunk
  ↓
BAAI/bge-reranker-v2-m3 重新評分與排序
  ↓
取重排後前 TOP_K 個 Chunk 並組成 Context
  ↓
qwen3:4b 根據 Context 產生回答
```

1. 使用 `PyPDFLoader` 解析上傳的 PDF；圖片檔或文字過少的 PDF 頁面會交給 OCR。
2. 啟用 VLM 時，`auto` 模式分析直接上傳圖片、需 OCR 的 PDF 頁與含內嵌圖片的 PDF 頁；`all` 模式分析全部頁面。同頁 OCR 與 VLM 共用一次 PDF 渲染。
3. VLM 結果會先通過 Pydantic Schema 驗證，並建立獨立 Visual Chunk；單頁失敗時保留可用的原生文字／OCR Chunk。
4. 保留原始檔名、頁碼、內容類型與解析方式，再依中文標點切分內容。
5. 透過 Ollama 的 `bge-m3` 為兩種 Chunk 產生 Embedding，並放入同一個 FAISS 索引。
6. 問答時先由 FAISS 取回 `RETRIEVAL_TOP_K` 個候選 Chunk。
7. 啟用 Reranker 時，以 `BAAI/bge-reranker-v2-m3` 重排候選，僅取前 `TOP_K` 個組成帶來源編號的 Context。
8. 由 Ollama 的 `qwen3:4b` 僅根據 Context 產生繁體中文回答；來源會區分「文件文字」與「視覺分析」。

Embedding 與 Reranker 的用途不同：Embedding 將問題與 Chunk 轉成向量，讓 FAISS 快速找出候選集合；Reranker 直接比較問題與候選文字，進行較精細但較耗時的第二階段排序。Reranker 由 Python 的 FlagEmbedding 載入 Hugging Face 模型，不透過 Ollama。

索引只存在目前的 Streamlit 工作階段，重新啟動應用程式後需重新上傳並建立知識庫。本版不會載入或寫出 FAISS pickle 檔案。

## 專案結構

```text
local-rag-assistant/
├── app.py
├── scripts/
│   ├── download_evaluation_pdfs.ps1
│   └── evaluate_retrieval.py
├── evaluation/
│   └── dataset.json
├── src/rag/
│   ├── config.py
│   ├── document_loader.py
│   ├── image_renderer.py
│   ├── visual_prompt.py
│   ├── vlm_service.py
│   ├── evaluation.py
│   ├── evaluation_cache.py
│   ├── reranker.py
│   ├── vector_store.py
│   ├── prompt.py
│   └── rag_service.py
├── tests/
│   └── test_reranker.py
├── documents/
│   └── .gitkeep
├── storage/
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

主要檔案職責：

- `app.py`：Streamlit 入口，負責畫面呈現、上傳流程、知識庫建立、問答操作與結果顯示。
- `src/rag/config.py`：讀取 `.env` 與環境變數，集中管理 Ollama、模型、Chunk 與檢索參數。
- `src/rag/document_loader.py`：協調 PDF／圖片的原生文字、OCR 與 VLM 處理，建立 Text／Visual Chunks。
- `src/rag/image_renderer.py`：共用 PDF 頁面渲染、EXIF 方向修正、色彩轉換與等比例縮圖。
- `src/rag/ocr_service.py`：使用 PaddleOCR 辨識 PDF 頁面與圖片文字。
- `src/rag/visual_prompt.py`：集中管理具防 Prompt Injection 規則的 VLM Prompt 與版本。
- `src/rag/vlm_service.py`：呼叫 Ollama VLM、驗證結構化輸出並格式化視覺描述。
- `src/rag/evaluation.py`：驗證 Golden Dataset、判定 Retrieval Hit，並計算 Hit Rate@1、@3、@5、@8 與 MRR。
- `src/rag/vector_store.py`：建立 Ollama Embedding client，將 Chunks 寫入 FAISS，並提供相似度搜尋。
- `src/rag/reranker.py`：延遲載入並快取 FlagEmbedding Reranker，負責第二階段評分與穩定排序。
- `src/rag/prompt.py`：集中管理系統提示詞與使用者提示詞格式，限制模型只能根據文件內容回答。
- `src/rag/rag_service.py`：協調檢索、Context 格式化、呼叫聊天模型與回傳回答來源。
- `tests/`：放置單元測試，涵蓋文件處理、Context 格式化與 Chunk 預覽分頁等邏輯。

## Windows 安裝與啟動

需要 Python 3.11 與已安裝的 [Ollama](https://ollama.com/)。以下指令請在專案根目錄的 PowerShell 執行。如果系統可使用 `py` 而沒有 `python`，請將下列指令中的 `python` 改成 `py -3.11`。

建立並啟用虛擬環境：

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

如果 PowerShell 阻擋啟用腳本，可執行：

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

安裝套件：

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

複製環境設定並視需要編輯：

```powershell
Copy-Item .env.example .env
```

若要使用視覺分析，需另外下載 VLM：

```powershell
ollama pull qwen3-vl:4b
```

可調整的環境參數：

```env
OLLAMA_BASE_URL=http://localhost:11434
EMBEDDING_MODEL=bge-m3
CHAT_MODEL=qwen3:4b
CHUNK_SIZE=500
CHUNK_OVERLAP=80
RERANKER_ENABLED=true
RERANKER_MODEL=BAAI/bge-reranker-v2-m3
RERANKER_USE_FP16=false
RETRIEVAL_TOP_K=20
TOP_K=4
OCR_MODE=auto
OCR_LANG=ch
OCR_MIN_TEXT_CHARS=30
OCR_DPI=300
OCR_ENABLE_IMAGES=true
VLM_ENABLED=false
VLM_MODEL=qwen3-vl:4b
VLM_MODE=auto
VLM_MAX_IMAGE_EDGE=1600
VLM_TIMEOUT_SECONDS=120
VLM_NUM_PREDICT=800
```

- `OLLAMA_BASE_URL`：Ollama 服務網址，預設為本機服務。
- `EMBEDDING_MODEL`：Embedding 模型名稱，用於將文字轉成語意向量。
- `CHAT_MODEL`：聊天模型名稱，用於根據檢索到的 Context 產生回答。
- `CHUNK_SIZE`：每個 Chunk 的目標字元數。
- `CHUNK_OVERLAP`：相鄰 Chunk 的重疊字元數，用於保留上下文連續性。
- `RERANKER_ENABLED`：是否啟用第二階段重排。設為 `false` 時不會載入 FlagEmbedding 或 Hugging Face 模型。
- `RERANKER_MODEL`：Hugging Face Reranker 模型名稱，預設 `BAAI/bge-reranker-v2-m3`。
- `RERANKER_USE_FP16`：是否使用 FP16；CPU 環境預設為 `false`。
- `RETRIEVAL_TOP_K`：FAISS 第一階段取回的候選 Chunk 數量，必須大於或等於 `TOP_K`。
- `TOP_K`：重排後真正提供給 Context 與 LLM 的 Chunk 數量。停用 Reranker 時則是 FAISS 直接取回數量。
- `OCR_MODE`：OCR 模式，可為 `auto`、`force` 或 `disabled`。自動模式只會 OCR 文字過少的 PDF 頁面。
- `OCR_LANG`：PaddleOCR 語言，預設 `ch`。
- `OCR_MIN_TEXT_CHARS`：自動模式下，單頁文字低於此字元數才執行 OCR。
- `OCR_DPI`：PDF 頁面轉圖片時使用的解析度。
- `OCR_ENABLE_IMAGES`：是否允許直接上傳 PNG/JPG 圖片做 OCR。
- `VLM_ENABLED`：是否啟用建庫階段的視覺分析；安全預設為 `false`。
- `VLM_MODEL`：Ollama VLM 名稱，預設 `qwen3-vl:4b`。
- `VLM_MODE`：`auto`、`all` 或 `disabled`。向量繪製的 PDF 圖表可能無法被 `auto` 偵測，此時請改用 `all`。
- `VLM_MAX_IMAGE_EDGE`：送入 VLM 前的圖片最長邊像素上限，不會放大原圖。
- `VLM_TIMEOUT_SECONDS`：單張圖片模型呼叫逾時秒數。
- `VLM_NUM_PREDICT`：VLM 結構化描述的最大輸出 Token 數。

介面中的「VLM 視覺分析」頁籤會顯示格式化描述、模型名稱與安全化的失敗原因。VLM 描述屬於模型生成內容，精確文字、金額、日期與型號仍應以原生文字／OCR 為主；本 MVP 不會在最終回答階段重新傳送原始圖片，也不永久保存圖片或 FAISS 索引。

若要使用 OCR，除 `requirements.txt` 之外，還需要依照本機 CPU/GPU 環境安裝 PaddlePaddle runtime。CPU 版可參考 PaddleOCR 官方安裝文件：

```powershell
python -m pip install paddlepaddle
```

下載本機模型（本專案不會自動下載）：

```powershell
ollama pull bge-m3
ollama pull qwen3:4b
```

不需要也不應執行 `ollama pull BAAI/bge-reranker-v2-m3`。Reranker 第一次啟用時會由 FlagEmbedding 自動下載至 Hugging Face 模型快取；下載時間取決於網路，CPU 載入與重排也會增加延遲及記憶體用量。

確認 Ollama 已啟動後執行：

```powershell
streamlit run app.py
```

## 執行測試

測試使用 Fake/Mock，不會連線到 Ollama，也不會下載模型：

```powershell
python -m pytest
```

也可進行 Python 語法檢查：

```powershell
python -m compileall app.py src tests scripts
```

## RAG Retrieval Evaluation

本專案提供一套與 Streamlit UI 完全分離的 Retrieval benchmark，用固定的
`documents/evaluation/*.pdf` 比較 FAISS Vector Similarity Search baseline 與 Reranker 重排結果。固定文件可讓每次調整參數後都使用相同語料，避免上傳內容不同而使實驗結果無法比較；正式問答流程仍只處理使用者在 Streamlit 上傳的 PDF，不會自動載入 benchmark 文件。

`evaluation/dataset.json` 是 Golden Dataset。每筆資料包含送進 Retriever 的 `question`、正確 Chunk 所屬的 `expected_source`，以及該 Chunk 必須包含的 `expected_text`。只有同一個取回的 Chunk 同時符合來源檔名與關鍵文字才算 Hit。

- **Hit Rate@1、@3、@5、@8、@RETRIEVAL_TOP_K**：依正確 Chunk 最早出現的排名，計算位於各排名門檻內的題數比例；最後一項使用完整 FAISS 候選數量。
- **MRR**（Mean Reciprocal Rank）：每題第一個正確 Chunk 排名的倒數平均；Rank 1 為 `1`、Rank 2 為 `0.5`，設定的 Top-K 中沒有正確 Chunk 則為 `0`。

Evaluation PDF 不會提交到 Git。第一次執行前，請從專案根目錄下載 NIST 官方文件並驗證 SHA-256：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/download_evaluation_pdfs.ps1
```

若檔案已存在且雜湊正確，腳本會直接跳過下載；若雜湊不同，只有在新檔案通過驗證後才會取代原檔。文件來源為 [NIST AI 100-1](https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-1.pdf) 與 [NIST AI 600-1](https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.600-1.pdf)。請保留 NIST 文件名稱與來源標示。

下載完成後，兩份固定文件會位於：

```text
documents/evaluation/
├── NIST.AI.100-1.pdf
└── NIST.AI.600-1.pdf
```

確認 Ollama 已啟動且已安裝 `.env` 指定的 Embedding 模型後，在專案根目錄執行：

```powershell
python scripts/evaluate_retrieval.py
```

每次執行時，Script 都會重新載入專案根目錄的 `.env`，顯示本次實際採用的設定，並先驗證 Ollama 連線與 Embedding 模型能否產生向量。環境檢查通過後才會解析 PDF 和建立索引，因此模型名稱錯誤或 Ollama 未啟動時會立即停止並顯示原因。

Script 會沿用正式 RAG 的 PDF parser、Chunking、Ollama Embedding、FAISS 建索引與 `similarity_search()`，但不會呼叫聊天模型。每題只執行一次 FAISS 搜尋：原始 `RETRIEVAL_TOP_K` 候選順序計為 Baseline，同一候選集合的完整重排順序計為 Reranked；不會先截斷成正式問答的 `TOP_K`。為了計算固定的 Hit Rate@1、@3、@5、@8，`RETRIEVAL_TOP_K` 必須至少為 8。正式判定仍以 `expected_source` 與 `expected_text` 的嚴格字串比對為準，不會做文字正規化、模糊或語意比對。

輸出分為 `Baseline Retrieval`、`Reranked Retrieval` 與 `Difference`。前兩部分各列 Hit Rate 與 MRR，Difference 列出重排後相對 Baseline 的指標差值。若 `RERANKER_ENABLED=false`，Evaluation 只執行並輸出 Baseline，不建立或驗證 Reranker，也不產生假造的 Reranked 指標。

正式評估前會先確認每題來源 PDF、解析全文中的 `expected_text`，以及來源正確的單一 Chunk 是否完整包含該文字。輸出會列出未通過題號及 `SOURCE_NOT_FOUND`、`EXPECTED_TEXT_NOT_IN_PARSED_PDF`、`EXPECTED_TEXT_NOT_IN_ANY_CHUNK` 等原因；可評估題目若正確 Chunk 未進入 Top-K，會記為 `GOLD_CHUNK_NOT_IN_TOP_K`。摘要包含通過檢查題數、各 MISS 原因、逐題 Result/Rank/Source/MISS Reason、Hit Rate 與以全部題目為分母的 MRR。

Evaluation 專用 FAISS 快取位於 `storage/retrieval_evaluation/<cache-key>/`，包含 `index.faiss`、`documents.json` 與 `manifest.json`。快取鍵由 PDF 檔名與內容 SHA-256，以及 Embedding 模型、Embedding 維度、Chunk Size、Chunk Overlap、切分符號、Metadata/頁碼設定、PDF loader 與相關解析套件版本共同決定；任一項變更都會自動改用新快取並重建。載入時也會驗證檔案雜湊、向量維度及 Chunk 數量，失敗時安全重建。這個磁碟快取只用於 Retrieval Evaluation，不改變 Streamlit 工作階段內索引的既有行為。

Reranker 設定不會加入 FAISS cache key，因為它不改變 PDF、Chunk、Embedding 維度或 FAISS index；切換 Reranker 開關或模型後可繼續使用既有 Evaluation FAISS cache。

## 測試問題建議

可上傳公開且可解析文字的 PDF 測試問答效果。請勿將下載的 PDF 加入 Git。

建立知識庫後，可在「Chunk 預覽」分頁依檔案、頁碼與關鍵字篩選切塊內容；送出問題後，也可展開「查看實際檢索到的原始文字區塊」確認回答依據。

可嘗試以下類型的問題：

- 文件主要在說明什麼？
- 文件列出哪些重點或風險？
- 文件是否提到特定名詞、流程或限制？
- 文件有沒有提供某個問題的明確答案？

若問題超出文件內容，系統應回答「根據目前收錄的文件，找不到足夠資訊回答這個問題。」而不是自行補充文件外資訊。

## 常見問題

- 無法連線：確認 Ollama 應用程式已啟動，預設服務網址是 `http://localhost:11434`。
- 找不到模型：重新執行上述兩個 `ollama pull` 指令。
- Reranker 第一次執行較久：首次使用會從 Hugging Face 下載模型並建立本機快取；後續會重用快取與目前程序中的模型實例。
- 無法載入 Reranker：確認已安裝 `FlagEmbedding`、網路可連線至 Hugging Face，並確認 `RERANKER_MODEL` 名稱正確。若本機記憶體有限，可在 `.env` 設定 `RERANKER_ENABLED=false` 暫時停用。
- PDF 沒有文字：確認 `OCR_MODE` 不是 `disabled`，且已安裝 PaddleOCR 與 PaddlePaddle。
- PDF 解壓縮錯誤：若看到 `Error -3 while decompressing data: invalid code lengths set`，通常代表 PDF 內部壓縮資料不標準或部分損毀。可先確認是否仍能建立知識庫；若無法解析，請用瀏覽器或 PDF 閱讀器開啟後「列印成 PDF」重新輸出一份再上傳。
- 回答不正確：展開「實際檢索到的原始文字區塊」，先確認檢索結果是否包含答案，再調整 Chunk 或問題用詞。
