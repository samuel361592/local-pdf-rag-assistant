# 本機 PDF RAG 問答助手

這是一個使用 Streamlit、Ollama 與 FAISS 製作的基礎 RAG 學習專案。使用者可上傳一份或多份公開 PDF，系統會從文件擷取文字、建立向量索引，再依檢索內容回答問題並標示來源檔名與頁碼。介面也提供 Chunk 預覽與問答即時流程，方便檢查系統實際檢索與回答的過程。

> 本專案僅供學習與資訊整理，不構成法律或合規建議。請勿上傳機密、個人資料或其他不應交由本機模型處理的內容。

## 使用技術

- Python：主要開發語言。
- Streamlit：建立本機 Web 操作介面，提供 PDF 上傳、知識庫建立、文件問答與 Chunk 預覽。
- LangChain：整合 RAG 流程中的文件載入、文字切分、Embedding、向量庫與聊天模型呼叫。
- PyPDFLoader：解析 PDF，將 PDF 頁面轉成 LangChain `Document`。
- RecursiveCharacterTextSplitter：將長文本依段落、換行與中文標點切成適合檢索的 Chunks。
- Ollama：在本機執行 Embedding 模型與聊天模型。
- bge-m3：Embedding 模型，負責將文字 Chunk 轉成語意向量。
- qwen3:4b：聊天模型，負責根據檢索到的文件內容產生繁體中文回答。
- FAISS：向量資料庫，用於儲存文字向量並執行相似度搜尋。
- python-dotenv：讀取 `.env` 設定檔。
- pytest：撰寫與執行單元測試。

## RAG 流程

```text
PDF 上傳
  ↓
PyPDFLoader 解析文字
  ↓
RecursiveCharacterTextSplitter 切分 Chunk
  ↓
bge-m3 產生 Embedding
  ↓
FAISS 建立向量索引
  ↓
使用者提問
  ↓
FAISS 取回相關 Chunk
  ↓
qwen3:4b 根據 Context 產生回答
```

1. 使用 `PyPDFLoader` 解析上傳的 PDF。
2. 保留原始檔名，並將頁碼轉成一基頁碼顯示，再依中文標點切分文字。
3. 透過 Ollama 的 `bge-m3` 產生 Embedding。
4. 將向量存入本次 Streamlit Session State 內的 FAISS 索引。
5. 問答時取回最相關的 Chunk，組成帶來源編號的 Context。
6. 由 Ollama 的 `qwen3:4b` 僅根據 Context 產生繁體中文回答。
7. 送出問題後，介面右側會顯示即時流程，包括檢索 Chunk、組合 Context、呼叫模型與回答完成狀態。

索引只存在目前的 Streamlit 工作階段，重新啟動應用程式後需重新上傳並建立知識庫。本版不會載入或寫出 FAISS pickle 檔案。

## 專案結構

```text
local-pdf-rag-assistant/
├── app.py
├── scripts/
│   ├── download_evaluation_pdfs.ps1
│   └── evaluate_retrieval.py
├── evaluation/
│   └── dataset.json
├── src/rag/
│   ├── config.py
│   ├── document_loader.py
│   ├── evaluation.py
│   ├── vector_store.py
│   ├── prompt.py
│   └── rag_service.py
├── tests/
├── documents/
│   └── evaluation/
│       ├── NIST.AI.100-1.pdf
│       └── NIST.AI.600-1.pdf
├── storage/
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

主要檔案職責：

- `app.py`：Streamlit 入口，負責畫面呈現、上傳流程、知識庫建立、問答操作與結果顯示。
- `src/rag/config.py`：讀取 `.env` 與環境變數，集中管理 Ollama、模型、Chunk 與檢索參數。
- `src/rag/document_loader.py`：處理 PDF 上傳內容、解析頁面文字、整理來源檔名與頁碼，並切分 Chunks。
- `src/rag/evaluation.py`：驗證 Golden Dataset、判定 Retrieval Hit，並計算 Hit Rate@1、@3、@5、@8 與 MRR。
- `src/rag/vector_store.py`：建立 Ollama Embedding client，將 Chunks 寫入 FAISS，並提供相似度搜尋。
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

可調整的環境參數：

```env
OLLAMA_BASE_URL=http://localhost:11434
EMBEDDING_MODEL=bge-m3
CHAT_MODEL=qwen3:4b
CHUNK_SIZE=500
CHUNK_OVERLAP=80
TOP_K=4
```

- `OLLAMA_BASE_URL`：Ollama 服務網址，預設為本機服務。
- `EMBEDDING_MODEL`：Embedding 模型名稱，用於將文字轉成語意向量。
- `CHAT_MODEL`：聊天模型名稱，用於根據檢索到的 Context 產生回答。
- `CHUNK_SIZE`：每個 Chunk 的目標字元數。
- `CHUNK_OVERLAP`：相鄰 Chunk 的重疊字元數，用於保留上下文連續性。
- `TOP_K`：問答時取回最相關的 Chunk 數量。

下載本機模型（本專案不會自動下載）：

```powershell
ollama pull bge-m3
ollama pull qwen3:4b
```

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
`documents/evaluation/*.pdf` 評估目前 FAISS Vector Similarity Search baseline。固定文件可讓每次調整參數後都使用相同語料，避免上傳內容不同而使實驗結果無法比較；正式問答流程仍只處理使用者在 Streamlit 上傳的 PDF，不會自動載入 benchmark 文件。

`evaluation/dataset.json` 是 Golden Dataset。每筆資料包含送進 Retriever 的 `question`、正確 Chunk 所屬的 `expected_source`，以及該 Chunk 必須包含的 `expected_text`。只有同一個取回的 Chunk 同時符合來源檔名與關鍵文字才算 Hit。

- **Hit Rate@1、@3、@5、@8、@TOP_K**：依正確 Chunk 最早出現的排名，計算位於各排名門檻內的題數比例；`@TOP_K` 使用 `.env` 設定值，若與固定門檻相同則只顯示一次。
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

Script 會沿用正式 RAG 的 PDF parser、Chunking、Ollama Embedding、FAISS 建索引與 `similarity_search()`，但不會呼叫聊天模型。Evaluation 依 `.env` 的 `TOP_K` 取回 Chunk；為了計算固定的 Hit Rate@1、@3、@5、@8，`TOP_K` 必須至少為 8。正式判定仍以 `expected_source` 與 `expected_text` 的嚴格字串比對為準，不會做文字正規化、模糊或語意比對。

正式評估前會先確認每題來源 PDF、解析全文中的 `expected_text`，以及來源正確的單一 Chunk 是否完整包含該文字。輸出會列出未通過題號及 `SOURCE_NOT_FOUND`、`EXPECTED_TEXT_NOT_IN_PARSED_PDF`、`EXPECTED_TEXT_NOT_IN_ANY_CHUNK` 等原因；可評估題目若正確 Chunk 未進入 Top-K，會記為 `GOLD_CHUNK_NOT_IN_TOP_K`。摘要包含通過檢查題數、各 MISS 原因、逐題 Result/Rank/Source/MISS Reason、Hit Rate 與以全部題目為分母的 MRR。

Evaluation 專用 FAISS 快取位於 `storage/retrieval_evaluation/<cache-key>/`，包含 `index.faiss`、`documents.json` 與 `manifest.json`。快取鍵由 PDF 檔名與內容 SHA-256，以及 Embedding 模型、Embedding 維度、Chunk Size、Chunk Overlap、切分符號、Metadata/頁碼設定、PDF loader 與相關解析套件版本共同決定；任一項變更都會自動改用新快取並重建。載入時也會驗證檔案雜湊、向量維度及 Chunk 數量，失敗時安全重建。這個磁碟快取只用於 Retrieval Evaluation，不改變 Streamlit 工作階段內索引的既有行為。

未來可用同一組固定 PDF 與 Golden Dataset 比較 Chunk Size、Chunk Overlap、Embedding Model、Hybrid Search 或 Reranker；本版僅實作既有 FAISS Vector Similarity Search baseline。

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
- PDF 沒有文字：掃描型 PDF 通常需要先用其他工具進行 OCR，本專案第一版不含 OCR。
- PDF 解壓縮錯誤：若看到 `Error -3 while decompressing data: invalid code lengths set`，通常代表 PDF 內部壓縮資料不標準或部分損毀。可先確認是否仍能建立知識庫；若無法解析，請用瀏覽器或 PDF 閱讀器開啟後「列印成 PDF」重新輸出一份再上傳。
- 回答不正確：展開「實際檢索到的原始文字區塊」，先確認檢索結果是否包含答案，再調整 Chunk 或問題用詞。
