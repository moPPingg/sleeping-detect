import os
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"

def count_data_sets():
    if not os.path.exists(DATA_DIR):
        print(f"[❌] LỖI: Không tìm thấy thư mục {DATA_DIR}")
        return

    print("Đang quét thư mục data (Tự động phát hiện cấu trúc của Train và Test)...")
    results = []
    total_train = 0
    total_test = 0

    for split in ['train', 'test']:
        split_dir = os.path.join(DATA_DIR, split)
        if not os.path.exists(split_dir):
            continue

        # Lấy các thư mục con ngay bên trong train/test
        subdirs = sorted([d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))])
        
        for subdir in subdirs:
            subdir_path = os.path.join(split_dir, subdir)
            children = sorted([d for d in os.listdir(subdir_path) if os.path.isdir(os.path.join(subdir_path, d))])
            
            # --- THUẬT TOÁN TỰ ĐỘNG THĂM DÒ ĐỘ SÂU ---
            # Xem thử bên trong có còn folder con (set) nào nữa không
            is_deep_structure = False
            for child in children:
                child_path = os.path.join(subdir_path, child)
                grand_children = [d for d in os.listdir(child_path) if os.path.isdir(os.path.join(child_path, d))]
                if len(grand_children) > 0:
                    is_deep_structure = True
                    break
            
            if is_deep_structure:
                # Cấu trúc giống TRAIN: Có thư mục bọc ngoài (VD: haar_cropped)
                variant = subdir
                for label in children:
                    label_path = os.path.join(subdir_path, label)
                    sets = [d for d in os.listdir(label_path) if os.path.isdir(os.path.join(label_path, d))]
                    count = len(sets)
                    
                    results.append({'Phân loại': split.upper(), 'Thư mục bọc': variant, 'Nhãn (Layer)': label, 'Số lượng Set': count})
                    if split == 'train': total_train += count
                    else: total_test += count
            else:
                # Cấu trúc giống TEST: Đi thẳng vào 3 nhãn luôn
                variant = "(Đi thẳng / Không có)"
                label = subdir
                sets = children # Ở đây children chính là các set
                count = len(sets)
                
                results.append({'Phân loại': split.upper(), 'Thư mục bọc': variant, 'Nhãn (Layer)': label, 'Số lượng Set': count})
                if split == 'train': total_train += count
                else: total_test += count

    # Tạo DataFrame từ list kết quả
    df = pd.DataFrame(results)
    
    # Tạo thêm các dòng tổng kết
    total_all = total_train + total_test
    summary_data = [
        {'Phân loại': '-----------------', 'Thư mục bọc': '-----------------', 'Nhãn (Layer)': '-----------------', 'Số lượng Set': '---'},
        {'Phân loại': 'TỔNG TRAIN', 'Thư mục bọc': '', 'Nhãn (Layer)': '', 'Số lượng Set': total_train},
        {'Phân loại': 'TỔNG TEST', 'Thư mục bọc': '', 'Nhãn (Layer)': '', 'Số lượng Set': total_test},
        {'Phân loại': 'TỔNG CỘNG (ALL)', 'Thư mục bọc': '', 'Nhãn (Layer)': '', 'Số lượng Set': total_all}
    ]
    summary_df = pd.DataFrame(summary_data)
    
    final_df = pd.concat([df, summary_df], ignore_index=True)
    
    # In ra màn hình console
    print("\n" + "="*75)
    print("📊 BÁO CÁO THỐNG KÊ SỐ LƯỢNG DỮ LIỆU (TỰ ĐỘNG ĐẾM CÙNG LÚC) 📊")
    print("="*75)
    print(final_df.to_string(index=False))
    print("="*75)
    
    # Xuất ra file CSV
    csv_filename = "data_summary_report.csv"
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = REPORTS_DIR / csv_filename
    final_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\n✅ Đã xuất báo cáo thành công ra file: {csv_filename}")

if __name__ == "__main__":
    count_data_sets()
