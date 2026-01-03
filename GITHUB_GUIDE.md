# 🚀 Hướng dẫn Push Code lên GitHub

## Bước 1: Tạo Repository trên GitHub

1. Đăng nhập vào [GitHub.com](https://github.com)
2. Click nút **"+"** ở góc trên bên phải → chọn **"New repository"**
3. Đặt tên repository (ví dụ: `drowsiness-detection`)
4. Chọn **Public** hoặc **Private**
5. **KHÔNG** tích vào "Initialize with README" (vì bạn đã có code rồi)
6. Click **"Create repository"**

## Bước 2: Add Remote và Push

### Nếu chưa có remote:
```bash
# Thay YOUR_USERNAME và REPO_NAME bằng thông tin của bạn
git remote add origin https://github.com/YOUR_USERNAME/REPO_NAME.git
git branch -M main
git push -u origin main
```

### Nếu đã có remote nhưng muốn đổi:
```bash
# Xem remote hiện tại
git remote -v

# Xóa remote cũ (nếu cần)
git remote remove origin

# Thêm remote mới
git remote add origin https://github.com/YOUR_USERNAME/REPO_NAME.git

# Push code
git push -u origin main
```

## Bước 3: Nếu dùng SSH thay vì HTTPS

```bash
# Thay đổi remote sang SSH
git remote set-url origin git@github.com:YOUR_USERNAME/REPO_NAME.git
git push -u origin main
```

## Lưu ý:

- Nếu branch của bạn là `master` thay vì `main`:
  ```bash
  git branch -M main  # Đổi tên branch thành main
  git push -u origin main
  ```

- Nếu gặp lỗi authentication, bạn cần:
  - Tạo Personal Access Token trên GitHub (Settings → Developer settings → Personal access tokens)
  - Dùng token thay vì password khi push

## Kiểm tra sau khi push:

```bash
git remote -v  # Xem remote đã được add chưa
git status    # Kiểm tra trạng thái
```

