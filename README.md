# MT-RAG Evaluation Pipeline (Windows Compatible)

Hướng dẫn cài đặt và chạy đánh giá RAG pipeline trên môi trường Windows. Dự án này được tối ưu hóa để chạy mà không cần các thư viện không tương thích với Windows như `flash-attn` hay `bitsandbytes`.

## 📋 Mục lục
- [Yêu cầu hệ thống](#yêu-cầu-hệ-thống)
- [Cài đặt](#cài-đặt)
- [Sử dụng](#sử-dụng)
  - [1. Tạo Predictions](#1-tạo-predictions)
  - [2. Chạy Đánh giá (Evaluation)](#2-chạy-đánh-giá-evaluation)
- [Cấu trúc dữ liệu](#cấu-trúc-dữ-liệu)
- [Xử lý lỗi thường gặp](#xử-lý-lỗi-thường-gặp)

---

## 💻 Yêu cầu hệ thống
- **OS**: Windows 10/11
- **Python**: 3.10 (Khuyến nghị 64-bit)
- **API**: Azure OpenAI hoặc OpenAI Key (để đánh giá)

---

## 🛠 Cài đặt

### 1. Tạo môi trường ảo
Mở PowerShell và chạy các lệnh sau để khởi tạo môi trường Python sạch:

```powershell
# Tạo venv với Python 3.10
py -3.10 -m venv venv

# Kích hoạt môi trường (Windows)
.\venv\Scripts\activate
```

### 2. Cài đặt dependencies
Cài đặt các thư viện tương thích với Windows từ file `requirements_win.txt`:

```powershell
pip install -r requirements_win.txt
```

> **Lưu ý**: File `requirements_win.txt` đã loại bỏ các thư viện chỉ chạy trên Linux/CUDA như `flash-attn`, `bitsandbytes`, `pytrec_eval` (bản gốc) để tránh lỗi build trên Windows.

---

## 🚀 Sử dụng

### 1. Tạo Predictions
Sử dụng script `generate_with_rag.py` để sinh câu trả lời từ RAG model của bạn.

```powershell
python scripts/evaluation/generate_with_rag.py `
  --input "C:\Users\Admin\Desktop\code\Final\mt-rag\mt-rag-benchmark-main\human\generation_tasks\RAG.jsonl" `
  --output "predictions\clapnq_test.jsonl" `
  --collection clapnq `
  --limit 10
```

### 2. Chạy Đánh giá (Evaluation)
Sử dụng **Azure OpenAI** (hoặc OpenAI) để chấm điểm kết quả. 

> ⚠️ **Quan trọng**: Trên PowerShell, sử dụng dấu backtick (`` ` ``) để xuống dòng.

```powershell
python scripts/evaluation/run_generation_eval.py `
  -i "predictions\clapnq_test.jsonl" `
  -o "outputs\rag_eval_output.jsonl" `
  -e "scripts/evaluation/config.yaml" `
  --provider openai `
  --openai_key "YOUR_OPENAI_API_KEY" `
  --azure_host "https://YOUR_RESOURCE_NAME.openai.azure.com"
```

**Quy trình đánh giá bao gồm:**
1.  **Algorithmic**: Tính điểm BLEU, ROUGE, BERTScore.
2.  **Ragas Faithfulness**: Kiểm tra độ trung thực (nếu dùng OpenAI).
3.  **IDK Judge**: Kiểm tra mô hình có biết từ chối khi không có thông tin hay không (dùng `gpt-4o-mini`).
4.  **RadBench Judge**: Đánh giá chất lượng câu trả lời tổng thể.

---

## 📂 Cấu trúc dữ liệu

File output (`predictions\*.jsonl`) phải tuân thủ định dạng JSONL sau để script đánh giá hoạt động chính xác:

```json
{
  "task_id": "unique_id_123",
  "input": [
    {
      "speaker": "user",
      "text": "Câu hỏi của người dùng?"
    }
  ],
  "targets": [
    {
      "speaker": "agent",
      "text": "Câu trả lời mẫu (Gold standard)."
    }
  ],
  "contexts": [
    "Đoạn văn bản retrieved 1...",
    "Đoạn văn bản retrieved 2..."
  ],
  "predictions": [
    {
      "text": "Câu trả lời do model sinh ra..."
    }
  ]
}
```

---

## ❓ Xử lý lỗi thường gặp

| Lỗi | Nguyên nhân | Cách khắc phục |
| :--- | :--- | :--- |
| `KeyError: 'targets'` | File prediction thiếu trường `targets`. | Cập nhật `generate_with_rag.py` để copy `targets` từ input sang output. |
| `KeyError: 'contexts'` | Prediction thiếu ngữ cảnh tìm kiếm. | Đảm bảo pipeline RAG ghi lại các văn bản đã retrieve vào trường `contexts`. |
| `PackageNotFoundError: bitsandbytes` | Cố gắng tải model 4-bit/8-bit của HF. | Trên Windows, không dùng `load_in_4bit=True`. Thay vào đó dùng model full precision hoặc API. |
| `Missing expression after unary operator --` | Lỗi cú pháp PowerShell. | Thay thế dấu `\` bằng dấu backtick (`` ` ``) khi xuống dòng lệnh dài. |
| `numpy build failed` | Xung đột phiên bản hoặc Python 32-bit. | Đảm bảo dùng Python 3.10 **64-bit** và `numpy==1.26.4`. |
