# 本機 PDF RAG 問答助手

這是一個使用 Streamlit、Ollama 與 FAISS 製作的基礎 RAG 學習專案。使用者可上傳一份或多份公開 PDF，系統會從文件擷取文字、建立向量索引，再依檢索內容回答問題並標示來源檔名與頁碼。

> 本專案僅供學習與資訊整理，不構成法律或合規建議。請勿上傳機密、個人資料或其他不應交由本機模型處理的內容。

## RAG 流程

1. 使用 `PyPDFLoader` 解析上傳的 PDF。
2. 保留原始檔名與一起算頁碼，並依中文標點切分文字。
3. 透過 Ollama 的 `bge-m3` 產生 Embedding。
4. 將向量存入本次 Streamlit Session State 內的 FAISS 索引。
5. 問答時取回最相關的 Chunk，組成帶來源編號的 Context。
6. 由 Ollama 的 `qwen3:4b` 僅根據 Context 產生繁體中文回答。

索引只存在目前的 Streamlit 工作階段，重新啟動應用程式後需重新上傳並建立知識庫。本版不會載入或寫出 FAISS pickle 檔案。

## 專案結構

```text
local-pdf-rag-assistant/
├── app.py
├── src/rag/
│   ├── config.py
│   ├── document_loader.py
│   ├── vector_store.py
│   ├── prompt.py
│   └── rag_service.py
├── tests/
├── documents/
├── storage/
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

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
python -m compileall app.py src tests
```

## 推薦測試文件與問題

可從行政院官方頁面取得「[行政院及所屬機關（構）使用生成式 AI 參考指引](https://www.ey.gov.tw/Page/448DE008087A1971/40c1a925-121d-4b6b-8f40-7e9e1a5401f2)」。請自行下載後於介面上傳，不要把 PDF 加入 Git。

可嘗試以下問題：

- 什麼是生成式人工智慧？
- 可以把個人資料輸入生成式 AI 嗎？
- 使用生成式 AI 產生內容時需要注意什麼？
- 公司應該購買哪一個 AI 產品？

最後一題通常不在指引的資訊範圍內，系統應回答「根據目前收錄的文件，找不到足夠資訊回答這個問題。」而不是自行推薦產品。

## 常見問題

- 無法連線：確認 Ollama 應用程式已啟動，預設服務網址是 `http://localhost:11434`。
- 找不到模型：重新執行上述兩個 `ollama pull` 指令。
- PDF 沒有文字：掃描型 PDF 通常需要先用其他工具進行 OCR，本專案第一版不含 OCR。
- 回答不正確：展開「實際檢索到的原始文字區塊」，先確認檢索結果是否包含答案，再調整 Chunk 或問題用詞。
