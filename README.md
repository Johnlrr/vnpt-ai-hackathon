# VNPT AI Hackathon 2025 - Track 2: The Builder
**Team Submission: Perplexity**

Repo này chứa mã nguồn giải pháp cho Track 2: The Builder. Chúng tôi xây dựng một hệ thống Advanced RAG Pipeline tập trung vào độ chính xác cao cho các câu hỏi Toán học và Kiến thức tổng hợp, sử dụng các kỹ thuật tiên tiến như Ensemble Retrieval, Reranking, Adaptive Temperature và Self-Correction (Reflexion).

## 1. Pipeline Flow
Hệ thống được thiết kế theo kiến trúc Modular RAG với luồng xử lý dữ liệu End-to-End như sau:
```text
graph TD
    A[Input Question] --> B{Safety Check}
    B -- Unsafe --> Z[Return Default Answer A]
    B -- Safe --> C[Query Expansion]
    
    subgraph Retrieval Phase
        C --> D1[Search General DB]
        C --> D2[Search Math/Logic DB]
        D1 & D2 --> E[Ensemble & Deduplication]
    end
    
    subgraph Optimization Phase
        E --> F[LLM Reranking]
        F --> G[Context Assembly]
    end
    
    subgraph Generation Phase
        G --> H{Classifier: Math vs General?}
        H -- Math --> I[Set Temp=0.05, Math Prompt]
        H -- General --> J[Set Temp=0.12, General Prompt]
        I & J --> K[Large LLM Generation (CoT)]
    end
    
    subgraph Verification Phase
        K --> L{Self-Correction Check}
        L -- Low Confidence / Math --> M[Reflexion: LLM Critique & Fix]
        L -- High Confidence --> N[Final Answer]
        M --> N
    end
    
    N --> O[Format Output CSV]
```
**Mô tả chi tiết các module**:
- Safety Guardrail: Kiểm tra câu hỏi đầu vào với Safety.index (Vector DB chứa các mẫu câu hỏi độc hại). Nếu khoảng cách vector < 0.35, hệ thống từ chối trả lời và trả về đáp án mặc định.

- Query Expansion (Biến thể câu hỏi): Sử dụng Small LLM để tạo ra các biến thể của câu hỏi gốc, giúp tăng khả năng tìm kiếm ngữ nghĩa đa chiều.

- Ensemble Retrieval: Thực hiện tìm kiếm song song trên 2 Vector Store riêng biệt (MathLogic và General Knowledge) để đảm bảo không bỏ sót kiến thức chuyên ngành.

- Reranking: Các tài liệu tìm được sẽ được chấm điểm lại (Rerank) bởi Small LLM để chọn ra top 7 tài liệu liên quan nhất, loại bỏ nhiễu.

- Adaptive Generation:

    - Temperature Tuning: Tự động điều chỉnh nhiệt độ sinh (temperature) dựa trên độ dài và loại câu hỏi (Toán học dùng temp thấp để chính xác, câu hỏi dài dùng temp cao hơn để sáng tạo).

    - Chain-of-Thought (CoT): Prompt ép buộc mô hình suy luận theo 4 bước: Hiểu đề -> Tìm công thức -> Tính toán -> Kết luận.

- Self-Correction (Cơ chế tự sửa lỗi): Với các câu hỏi Toán hoặc khi mô hình đưa ra lý giải quá ngắn/không chắc chắn, hệ thống kích hoạt Large LLM đóng vai trò "người phản biện" (Critique) để kiểm tra lại logic và sửa lỗi nếu cần.

## 2. Data Processing
Quy trình xử lý dữ liệu từ Raw Text sang Vector Database được thực hiện và đóng gói sẵn trong thư mục assets/.

Các bước thực hiện:
**Data Collection & Cleaning** :

Tổng hợp dữ liệu từ các nguồn tài liệu Toán học, Vật lý, Tâm lý học, Sinh học, Kinh tế và nhiều kiến thức chung khác...

Loại bỏ các ký tự nhiễu, chuẩn hóa công thức toán học (LaTeX/Unicode).

**Chunking (Phân mảnh)**:

Sử dụng chiến lược Recursive Character Splitter.

Chunk Size: 1800 tokens (đảm bảo giữ trọn vẹn ngữ cảnh một đoạn văn/bài toán).

Overlap: 200 tokens (để giữ tính liên kết giữa các đoạn).

**Embedding**:

Sử dụng API vnptai_hackathon_embedding.

Toàn bộ chunks được chuyển đổi thành vector float.

**Indexing (Lưu trữ)**:

Vector được đánh chỉ mục bằng thư viện FAISS (Facebook AI Similarity Search) để tối ưu tốc độ truy vấn.

Metadata (nội dung text gốc) được lưu trữ song song trong file .pkl.

## 3. Resource Initialization
Để chạy được pipeline, cần khởi tạo môi trường và tài nguyên theo hướng dẫn sau.

### 3.1. Cấu trúc thư mục
Repository này đã được tổ chức theo chuẩn Docker mount:
```text
.
├── assets/                     # Chứa Vector DB & Metadata (Đã pre-build)
│   ├── Safety.index
│   ├── vnpt_mathlogic_knowledge.index
│   ├── vnpt_knowledge_MERGED.index
│   └── *.pkl files
├── Dockerfile                  # Cấu hình môi trường
├── predict.py                  # Entry-point của Pipeline
├── inference.sh                # Script chạy chính
└── requirements.txt            # Danh sách thư viện
```
### 3.2. Yêu cầu hệ thống & Thư viện
- OS: Linux (Ubuntu 20.04 recommended) hoặc Docker Environment.

- Python: 3.8+

- Các thư viện chính (requirements.txt):

    - faiss-cpu==1.7.4: Truy vấn vector tốc độ cao.

    - requests: Gọi API LLM/Embedding.

    - numpy, pandas: Xử lý dữ liệu ma trận và CSV.

    - tqdm: Hiển thị tiến trình.

### 3.3. Hướng dẫn Build & Run (Docker)
**Bước 1: Cấu hình API Key**
Trước khi build, đảm bảo các biến môi trường (API Tokens) đã được thiết lập trong file predict.py hoặc truyền qua Docker run (khuyến nghị điền trực tiếp vào predict.py để đơn giản hóa quá trình chấm thi).

**Bước 2: Build Docker Image**
```text
sudo docker build -t Perplexity .
```

**Bước 3: Run Container (Giả lập môi trường chấm thi)**
Hệ thống sẽ mount file private_test.json vào thư mục /code trong container.
```text
sudo docker run --rm \
  -v /path/to/your/data:/code \
  [Tên_Docker_Image_Của_Bạn]
```

**Bước 4: Kiểm tra kết quả**
Sau khi chạy xong, file submission.csv sẽ được sinh ra tại thư mục /path/to/your/data (được map từ /code/submission.csv).

**4. Cam kết**
Chúng tôi cam kết mã nguồn này là sản phẩm trí tuệ của đội thi, tuân thủ mọi quy định của BTC VNPT AI Hackathon 2024.

Created by team Perplexity - Dec 2025