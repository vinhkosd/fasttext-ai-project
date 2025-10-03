# FastText AI Project + Duckling

Dự án demo phân loại intent bằng **FastText** và trích xuất thời gian bằng **Duckling** (Facebook).

## 🚀 Yêu cầu hệ thống
- Python 3.9+ (khuyên dùng 3.10)
- Git
- Docker (để chạy Duckling, dễ hơn so với build từ source)

## 📦 Cài đặt

### 1. Clone repo & tạo virtualenv
```bash
git clone https://github.com/vinhkosd/fasttext-ai-project.git
cd fasttext-ai-project
```
# tạo venv (nếu dùng Python 3.9)
```
python3 -m venv venv
source venv/bin/activate   # macOS/Linux
```
# hoặc: 
```
.\venv\Scripts\activate  # Windows PowerShell
```
### 3. Chạy Duckling bằng Docker Compose
```
docker-compose up -d
```