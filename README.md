✅ 1. Tạo môi trường Python 3.10 sạch
1.1. Tạo venv mới
py -3.10 -m venv venv

1.2. Kích hoạt
.\venv\Scripts\activate

✅ 2. Cài đặt các package tương thích Windows

Vì flash-attn + bitsandbytes KHÔNG chạy trên Windows → phải dùng bản requirements Win-safe.

Tạo file:

scripts/evaluation/requirements_win.txt
numpy==1.26.4
pandas
certifi
tqdm
beautifulsoup4
lxml
evaluate==0.4.3
bert_score
rouge-score
protobuf==5.29.3
ragas==0.1.9
langchain==0.1.20
langchain-community==0.2.6
pydantic==2.11.7

Cài đặt
pip install -r scripts/evaluation/requirements_win.txt


✔ Không có bitsandbytes
✔ Không có flash-attn
✔ Không có pytrec_eval
✔ Tương thích Python 3.10
✔ Tương thích Windows

✅ 3. Tạo prediction file từ mô hình RAG của bạn

Bạn đã chạy được:

python scripts/evaluation/generate_with_rag.py `
  --input human/generation_tasks/RAG.jsonl `
  --output predictions.jsonl `
  --limit 1


File output đúng chuẩn phải có format như:

{
  "task_id": "xxx",
  "input": [...],
  "targets": [...],
  "contexts": [...],
  "predictions": [...]
}


Nếu pipeline của bạn không sinh "contexts" hoặc "targets" → phải bổ sung trong generate_with_rag.py.

🔧 4. Chọn chế độ đánh giá: OpenAI / Azure (khuyên dùng)
Vì bạn đang dùng Windows → KHÔNG dùng local HuggingFace model để evaluate.

Thay thế:

--provider openai

✅ 5. Chạy đánh giá bằng Azure OpenAI
⚠ PowerShell KHÔNG dùng dấu \ để xuống dòng

Bạn phải dùng backtick (`)

Ví dụ:

python scripts/evaluation/run_generation_eval.py `
  -i predictions.jsonl `
  -o outputs/rag_eval_output.jsonl `
  -e scripts/evaluation/config.yaml `
  --provider openai `
  --judge_model gpt-4o-mini `
  --openai_key "<YOUR_KEY>" `
  --azure_host "https://<your-resource>.cognitiveservices.azure.com/"

⚠ LƯU Ý BẢO MẬT

🔥 Không bao giờ paste API key lên Internet. Hãy tạo key mới ngay!

✅ 6. Khi chạy, sẽ diễn ra các bước sau
6.1. Algorithmic evaluation

– BLEU/ROUGE/BERTScore
– Ragas Faithfulness (nếu dùng OpenAI)

6.2. IDK judge

– Dùng gpt-4o-mini để đánh giá "biết/không biết".

6.3. RadBench judge

– Đánh giá độ phù hợp câu trả lời.

6.4. Xuất file kết quả

→ outputs/rag_eval_output.jsonl

✅ 7. Các lỗi phổ biến và cách xử lý
Lỗi	Nguyên nhân	Cách sửa
KeyError: 'targets'	File prediction thiếu targets	Sửa generate_with_rag.py để sinh targets
KeyError: 'contexts'	Prediction không có ngữ cảnh	Thêm list context vào file
PackageNotFoundError: bitsandbytes	HF model muốn load 4bit	Không dùng HF model, dùng provider=openai
Missing expression after unary operator --	PowerShell lỗi xuống dòng	Dùng backtick thay vì`
numpy 1.26.4 build failed	Bạn đang dùng Python 32-bit	Cần Python 3.10 64-bit
🔥 8. Mẫu prediction đúng chuẩn (để benchmark hoạt động)
{
  "task_id": "abc123<::>1",
  "input": [
    {
      "speaker": "user",
      "text": "where do the arizona cardinals play this week"
    }
  ],
  "targets": [
    {
      "speaker": "agent",
      "text": "I'm sorry, but I don't have the answer to your question."
    }
  ],
  "contexts": ["<your retrieved chunk 1>", "<chunk 2>"],
  "predictions": [
    {
      "text": "Xin chào, tôi không có thông tin về lịch thi đấu…"
    }
  ]
}

🎉 9. Tóm tắt pipeline chuẩn cho Windows
Bước 1: Python 3.10 64-bit
Bước 2: Venv sạch
Bước 3: Cài requirements_win.txt
Bước 4: Sinh predictions.jsonl
Bước 5: Chạy benchmark bằng Azure OpenAI

🔥 Không dùng bitsandbytes, flash-attn, HF local model trên Windows.
