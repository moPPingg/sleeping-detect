# 📝 GIT IGNORE - HƯỚNG DẪN CHI TIẾT

## 🎯 CÁC CÂU HỎI THƯỜNG GẶP

---

## 1️⃣ LÀM SAO GIT BIẾT FILE NÀO KHÔNG PUSH?

### **Trả lời: File `.gitignore`**

Git sử dụng file `.gitignore` ở root folder để biết file/folder nào cần **bỏ qua**.

### **Cách hoạt động:**

```
📁 your-project/
├── .gitignore          ← File này nói cho Git biết bỏ qua gì
├── source.py           ← Sẽ được push (không có trong .gitignore)
├── data.csv            ← KHÔNG push (có trong .gitignore)
└── model.pkl           ← KHÔNG push (có trong .gitignore)
```

### **Ví dụ `.gitignore` của bạn:**

```gitignore
# Bỏ qua tất cả file CSV
*.csv
face_data.csv

# Bỏ qua tất cả file PKL (model files)
*.pkl
drowsiness_model.pkl
scaler.pkl

# Bỏ qua file MediaPipe
face_landmarker.task

# Bỏ qua ảnh generated
*.png
```

### **Khi chạy `git add .`:**

```bash
git add .  # Thêm TẤT CẢ files

# Git sẽ TỰ ĐỘNG:
✓ ADD:    source.py          (không có trong .gitignore)
✓ ADD:    README.md          (không có trong .gitignore)
✗ SKIP:   data.csv           (match với *.csv trong .gitignore)
✗ SKIP:   model.pkl          (match với *.pkl trong .gitignore)
✗ SKIP:   face_landmarker.task (có trong .gitignore)
```

**→ Bạn KHÔNG CẦN xóa manually! Git tự động bỏ qua!**

---

## 2️⃣ TẠI SAO KHÔNG PUSH FILE PKL?

### **Lý do 1: Quá lớn cho GitHub**

```
drowsiness_model.pkl:     ~150 MB   ❌ Vượt limit
scaler.pkl:               ~10 KB    (nhỏ nhưng không cần)
face_landmarker.task:     ~300 MB   ❌ KHÔNG THỂ push!

GitHub giới hạn:
- File đơn:               100 MB
- Repository:             1-5 GB (tùy account)
```

**→ File quá lớn sẽ bị REJECT khi push!**

### **Lý do 2: Có thể tái tạo**

```bash
# User clone về và TỰ TẠO model:
python model_trainer.py   # → Tạo drowsiness_model.pkl
                         # → Tạo scaler.pkl
                         # → Chỉ mất 2-3 phút!
```

**→ KHÔNG CẦN push vì user tự train được!**

### **Lý do 3: Bảo mật và Tính học tập**

- ✅ User tự train → Học được cách hoạt động
- ✅ User có data riêng → Model phù hợp hơn
- ✅ Không share model → Tránh vấn đề bản quyền

---

## 3️⃣ NGƯỜI KHÁC CLONE VỀ CÓ CHẠY ĐƯỢC KHÔNG?

### **Trả lời: CÓ! Nhưng cần setup trước**

### **Kịch bản 1: Clone repository của bạn**

```bash
# User khác clone về
git clone https://github.com/yourusername/driver-monitoring-system.git
cd driver-monitoring-system
```

**Họ sẽ có:**
```
✓ Source code (.py files)          - CÓ
✓ Documentation (README, etc.)     - CÓ
✓ Notebook (project_documentation) - CÓ
✓ requirements.txt                 - CÓ
✓ haarcascade XML                  - CÓ

✗ face_data.csv                    - KHÔNG
✗ drowsiness_model.pkl             - KHÔNG
✗ scaler.pkl                       - KHÔNG
✗ face_landmarker.task             - KHÔNG
```

### **Họ CẦN LÀM GÌ để chạy được?**

#### **BƯỚC 1: Cài dependencies**
```bash
pip install -r requirements.txt
```

#### **BƯỚC 2: Download MediaPipe model**
```bash
# Option A: Download từ MediaPipe official
wget https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task

# Option B: Tự download và đặt vào project folder
# Link: https://developers.google.com/mediapipe/solutions/vision/face_landmarker
```

#### **BƯỚC 3: Thu thập data**
```bash
python data_collector.py
# → Quay video, collect ~600 samples mỗi class
# → Tạo face_data.csv
# → Mất ~10-15 phút
```

#### **BƯỚC 4: Train model**
```bash
python model_trainer.py
# → Đọc face_data.csv
# → Train Random Forest
# → Tạo drowsiness_model.pkl và scaler.pkl
# → Mất ~2-3 phút
```

#### **BƯỚC 5: Chạy hệ thống**
```bash
python drowsiness_detection_system.py
# → System chạy với model vừa train!
```

---

## 4️⃣ LÀM SAO ĐỂ DỄ DÀNG HƠN CHO USER?

### **Giải pháp: Viết hướng dẫn rõ ràng trong README**

Tôi đã tạo sẵn hướng dẫn chi tiết trong `README.md` và `QUICK_START.md`:

```markdown
## Quick Start

### Prerequisites
- Python 3.8+
- Webcam
- ~2GB free disk space

### Installation

1. **Clone repository:**
   ```bash
   git clone https://github.com/yourusername/driver-monitoring-system.git
   cd driver-monitoring-system
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download MediaPipe model:**
   - Download from: https://storage.googleapis.com/.../face_landmarker.task
   - Place in project root folder

4. **Collect training data:**
   ```bash
   python data_collector.py
   ```
   Follow instructions to collect ~600 samples per class.

5. **Train model:**
   ```bash
   python model_trainer.py
   ```
   Wait ~2 minutes for training to complete.

6. **Run system:**
   ```bash
   python drowsiness_detection_system.py
   ```
```

---

## 5️⃣ NẾU MUỐN SHARE MODEL FILES?

### **Option A: Git LFS (Git Large File Storage)**

```bash
# Cài Git LFS
git lfs install

# Track large files
git lfs track "*.pkl"
git lfs track "face_landmarker.task"

# Commit và push
git add .gitattributes
git add drowsiness_model.pkl scaler.pkl
git commit -m "Add model files with LFS"
git push
```

**Ưu điểm:**
- ✅ User clone về có sẵn model
- ✅ Chạy ngay được

**Nhược điểm:**
- ⚠️ GitHub LFS free: 1GB storage, 1GB bandwidth/month
- ⚠️ Clone chậm hơn
- ⚠️ Phức tạp hơn

### **Option B: Google Drive / Dropbox**

Trong README.md:

```markdown
## Pre-trained Models (Optional)

If you don't want to train yourself, download pre-trained models:

📦 [Download from Google Drive](https://drive.google.com/file/d/...)

**Contents:**
- drowsiness_model.pkl (150 MB)
- scaler.pkl (10 KB)
- face_landmarker.task (300 MB)

**Installation:**
1. Download and extract
2. Place files in project root
3. Run: `python drowsiness_detection_system.py`
```

### **Option C: GitHub Releases**

```
1. Go to GitHub repo → Releases
2. Create new release (v1.0)
3. Upload model files as assets
4. User download from releases page
```

---

## 6️⃣ KIỂM TRA .GITIGNORE CÓ HOẠT ĐỘNG?

### **Cách 1: Check trước khi commit**

```bash
# Xem files sẽ được add
git status

# Nếu thấy file KHÔNG MUỐN push:
git status | grep "face_data.csv"    # Không thấy = OK!
git status | grep "model.pkl"        # Không thấy = OK!
```

### **Cách 2: Check files đã push**

```bash
# List files trong Git
git ls-files

# Nếu THẤY file không muốn:
git ls-files | grep ".pkl"    # Nếu có kết quả = BAD!
```

### **Cách 3: Xem trên GitHub**

```
1. Vào repository trên GitHub
2. Browse files
3. Kiểm tra KHÔNG thấy:
   - .csv files
   - .pkl files
   - .task files
   - .png files (generated)
```

---

## 7️⃣ NẾU ĐÃ PUSH NHẦM FILE KHÔNG MUỐN?

### **Bước 1: Xóa file khỏi Git (giữ local)**

```bash
# Xóa file khỏi Git nhưng GIỮ file local
git rm --cached face_data.csv
git rm --cached drowsiness_model.pkl
git rm --cached "*.pkl"

# Commit
git commit -m "Remove large files from Git"

# Push
git push origin main
```

### **Bước 2: Thêm vào .gitignore**

```bash
# Đảm bảo .gitignore có dòng này:
echo "*.pkl" >> .gitignore
echo "*.csv" >> .gitignore

git add .gitignore
git commit -m "Update .gitignore"
git push
```

### **Bước 3: Clean history (nếu cần)**

```bash
# Xóa hoàn toàn khỏi Git history
git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch drowsiness_model.pkl" \
  --prune-empty --tag-name-filter cat -- --all

# Force push
git push origin --force --all
```

---

## 8️⃣ BEST PRACTICES

### **✅ DO (Nên làm):**

```
✓ Push source code
✓ Push documentation
✓ Push requirements.txt
✓ Push .gitignore
✓ Push small config files (< 1MB)
✓ Hướng dẫn user tự tạo model
```

### **❌ DON'T (Không nên):**

```
✗ Push data files (CSV, JSON > 10MB)
✗ Push model files (PKL, H5, PTH)
✗ Push large libraries (node_modules, venv)
✗ Push sensitive data (API keys, passwords)
✗ Push generated files (can regenerate)
✗ Push binary files (unless necessary)
```

### **📝 README Template:**

```markdown
## Setup Instructions

**IMPORTANT:** This repository does NOT include:
- Training data (`face_data.csv`)
- Trained models (`drowsiness_model.pkl`, `scaler.pkl`)
- MediaPipe model (`face_landmarker.task`)

You need to:
1. Download MediaPipe model (link provided)
2. Collect your own data (~15 minutes)
3. Train the model (~2 minutes)

See [QUICK_START.md](QUICK_START.md) for step-by-step guide.
```

---

## 📊 TÓM TẮT

| Câu hỏi | Trả lời |
|---------|---------|
| **Git biết bỏ qua file nào?** | Từ file `.gitignore` |
| **Tại sao không push PKL?** | Quá lớn (>100MB), có thể tái tạo |
| **User clone về chạy được không?** | Có, sau khi setup (5 bước) |
| **Bao lâu để setup?** | ~20 phút (download + collect + train) |
| **Có cách nào dễ hơn?** | Share model qua Google Drive/LFS |
| **Đã push nhầm thì sao?** | `git rm --cached filename` |

---

## 🎯 KẾT LUẬN

### **Chiến lược hiện tại (Recommended):**

✅ **PUSH:** Source code, docs, notebook  
❌ **KHÔNG PUSH:** Data, models, generated files  
📝 **HƯỚNG DẪN:** User tự collect & train (~20 phút)

**Ưu điểm:**
- ✅ Repository nhẹ (~23 MB)
- ✅ Clone nhanh
- ✅ Professional
- ✅ User học được cách hoạt động
- ✅ Không vi phạm GitHub limits

**Nhược điểm:**
- ⚠️ User cần setup trước khi chạy
- ⚠️ Mất ~20 phút setup lần đầu

### **Alternative (Nếu muốn dễ hơn):**

Upload models lên Google Drive, thêm link vào README:

```markdown
## Quick Start (With Pre-trained Models)

**Download pre-trained models:** [Google Drive](link)

1. Install dependencies: `pip install -r requirements.txt`
2. Download and extract models to project folder
3. Run: `python drowsiness_detection_system.py`

**Setup time:** 5 minutes ⚡
```

---

<div align="center">

## ✅ .GITIGNORE ĐANG HOẠT ĐỘNG TỐT!

**Các file lớn đã được bỏ qua tự động**  
**Repository của bạn clean và professional**  
**Users có thể clone và setup dễ dàng**

🎉 **Perfect Setup!** 🎉

</div>

---

**Tóm tắt ngắn gọn:**
1. `.gitignore` → Git tự động bỏ qua files trong đó
2. Không push PKL → Quá lớn, user tự train được
3. User clone về → Cần setup (20 phút) nhưng chạy được
4. Đã có hướng dẫn chi tiết trong README
5. Nếu muốn dễ hơn → Share models qua Google Drive

