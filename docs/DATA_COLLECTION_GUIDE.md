# Auto Drowsy Data Collector

## Mục đích
Thu thập dữ liệu **ngủ gật (Drowsy/Microsleep)** tự động để tăng dataset lên **20,000-30,000 samples**.

## Tính năng

### 1. **Tự động phát hiện**
- ✅ Tự động tính EAR (Eye Aspect Ratio)
- ✅ Phân biệt Drowsy vs Microsleep
- ✅ Không cần nhấn phím thủ công

### 2. **Smart Detection**
- **Drowsy**: EAR < 0.25, giữ ít nhất 0.5s (15 frames)
- **Microsleep**: EAR < 0.20, giữ ít nhất 1.5s (45 frames)

### 3. **Progress Tracking**
- Hiển thị số samples thu thập real-time
- Progress bar trực quan
- Auto-save mỗi 30 giây

### 4. **Data Safety**
- Tự động merge với data cũ
- Không mất dữ liệu khi thoát giữa chừng

## Cách sử dụng

### Bước 1: Chạy tool
```bash
python auto_collect_drowsy.py
```

### Bước 2: Làm theo hướng dẫn
1. **Ngồi trước camera** (khoảng cách 50-70cm)
2. **Nhắm mắt lại** và giữ nguyên
3. Tool sẽ tự động detect và lưu data
4. Nhấn **'q'** để thoát bất cứ lúc nào

### Bước 3: Kiểm tra kết quả
```bash
python -c "import pandas as pd; df = pd.read_csv('face_data.csv'); print(f'Total: {len(df)} samples'); print(df['label'].value_counts())"
```

## Các trạng thái

| Trạng thái | Ý nghĩa | Màu |
|-----------|---------|-----|
| `Eyes Open` | Mắt mở, chưa thu thập | 🟢 Xanh |
| `Detecting...` | Đang phát hiện (đếm frames) | 🟡 Vàng |
| `DROWSY DETECTED!` | Thu thập Drowsy | 🟠 Cam |
| `MICROSLEEP DETECTED!` | Thu thập Microsleep | 🔴 Đỏ |

## Tips để thu thập hiệu quả

### 1. **Thay đổi tư thế**
- Ngồi thẳng, ngả lưng, nghiêng đầu
- Đa dạng góc nhìn giúp model robust hơn

### 2. **Thay đổi ánh sáng**
- Sáng mạnh, yếu, từ bên cạnh
- Giúp model hoạt động tốt ở mọi điều kiện

### 3. **Thay đổi thời gian nhắm mắt**
- Nhắm nhanh (0.5-1s) → Drowsy
- Nhắm lâu (1.5-3s) → Microsleep
- Cân bằng giữa 2 loại

### 4. **Thu thập nhiều session**
- Mỗi session 10-15 phút
- Nghỉ 5 phút giữa các session
- Tổng 5-10 sessions để đủ 20-30k samples

## Thông số kỹ thuật

| Tham số | Giá trị | Ghi chú |
|---------|---------|---------|
| `TARGET_SAMPLES` | 25,000 | Mục tiêu mặc định |
| `EAR_THRESHOLD_DROWSY` | 0.25 | Ngưỡng Drowsy |
| `EAR_THRESHOLD_MICROSLEEP` | 0.20 | Ngưỡng Microsleep |
| `MIN_FRAMES_DROWSY` | 15 | ~0.5 giây |
| `MIN_FRAMES_MICROSLEEP` | 45 | ~1.5 giây |
| `SAVE_INTERVAL` | 30s | Tự động lưu |

## Ước tính thời gian

| Mục tiêu | Thời gian ước tính |
|----------|-------------------|
| 5,000 samples | ~2-3 giờ |
| 10,000 samples | ~4-6 giờ |
| 20,000 samples | ~8-12 giờ |
| 30,000 samples | ~12-18 giờ |

**Lưu ý**: Thời gian thực tế phụ thuộc vào tần suất nhắm mắt và thời gian giữ.

## Troubleshooting

### ❌ "Cannot open camera"
```bash
# Thử các camera khác
cap = cv2.VideoCapture(1)  # hoặc 2, 3
```

### ❌ "No face detected"
- Đảm bảo đủ ánh sáng
- Ngồi gần camera hơn
- Đảm bảo mặt không bị che khuất

### ❌ "EAR không đủ thấp"
- Nhắm chặt mắt hơn
- Điều chỉnh `EAR_THRESHOLD_DROWSY` trong code

## Kiểm tra chất lượng data

```python
import pandas as pd

df = pd.read_csv('face_data.csv')

print(f"Total samples: {len(df)}")
print("\nClass distribution:")
print(df['label'].value_counts())
print("\nPercentage:")
print(df['label'].value_counts(normalize=True) * 100)
```

**Lý tưởng**: Mỗi class ~25% (balanced dataset)

## Tối ưu hóa

### Tăng tốc thu thập
```python
MIN_FRAMES_DROWSY = 10      # Giảm từ 15 → 10
MIN_FRAMES_MICROSLEEP = 30  # Giảm từ 45 → 30
```

### Chất lượng cao hơn
```python
MIN_FRAMES_DROWSY = 20      # Tăng lên 20
MIN_FRAMES_MICROSLEEP = 60  # Tăng lên 60
```

## Sau khi thu thập xong

### 1. Kiểm tra data
```bash
python -c "import pandas as pd; print(pd.read_csv('face_data.csv').info())"
```

### 2. Train lại model
```bash
python train_model.py
```

### 3. Test với data mới
```bash
python drowsiness_detection_system.py
```

## Lưu ý quan trọng

⚠️ **Không xóa file `face_data.csv`** trong quá trình thu thập  
⚠️ **Backup data thường xuyên**: `cp face_data.csv face_data_backup.csv`  
⚠️ **Kiểm tra disk space**: 30k samples ≈ 100-150 MB

---

**Happy Data Collecting! 🎯**

