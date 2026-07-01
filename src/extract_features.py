import os
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import models
from PIL import Image
import numpy as np
from dataclasses import dataclass
from typing import List
from tqdm import tqdm

# GPU Memory Optimization
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.set_per_process_memory_fraction(0.95)  # Use 95% of GPU VRAM

class Config:
    def __init__(self):
        self.PROJECT_ROOT = Path(__file__).resolve().parent.parent
        self.TRAIN_DIR = self.PROJECT_ROOT / "data" / "train"
        self.TEST_DIR = self.PROJECT_ROOT / "data" / "test"
        self.OUT_DIR = self.PROJECT_ROOT / "features"
        
        # Tạo thư mục output nếu chưa tồn tại
        os.makedirs(self.OUT_DIR, exist_ok=True)
        
        self.out_lstm = self.OUT_DIR / "X_seq.npy"
        self.out_rf = self.OUT_DIR / "X_flat.npy"
        self.out_labels = self.OUT_DIR / "y.npy"
        
        self.batch_size: int = 30  # Số frame mỗi video (dữ liệu có 30 ảnh/folder)  
        self.valid_exts: tuple = ('.jpg', '.jpeg', '.png', '.bmp')
        self.labels = {"sleep": 0, "awake": 1, "microsleep": 2}
        
        self.data_roots: List[Path] = []
        
        # Áp dụng logic quét thông minh cho CẢ TRAIN VÀ TEST
        for target_dir in [self.TRAIN_DIR, self.TEST_DIR]:
            if os.path.exists(target_dir):
                # Lấy danh sách các thư mục con bên trong
                subfolders = [d.lower() for d in os.listdir(target_dir) if os.path.isdir(os.path.join(target_dir, d))]
                
                # Trường hợp 1: Nếu mở thư mục ra thấy ngay các nhãn (awake, sleep, microsleep)
                if any(label in subfolders for label in self.labels.keys()):
                    self.data_roots.append(target_dir)
                # Trường hợp 2: Nếu mở ra thấy thư mục trung gian (như haar_cropped, train_augmented...)
                else:
                    for d in os.listdir(target_dir):
                        full_path = os.path.join(target_dir, d)
                        if os.path.isdir(full_path):
                            self.data_roots.append(full_path)


class FeatureExtractor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device_name = ('GPU' if self.device.type == 'cuda' else 'CPU')
        print(f"[{device_name}] Loading EfficientNet-B2...")
        
        # Load Model
        weights = models.EfficientNet_B2_Weights.DEFAULT
        self.model = models.efficientnet_b2(weights=weights)
        self.model.classifier = nn.Sequential()
        self.model.to(self.device)
        self.model.eval()
        
        # Debug: verify model is on correct device
        if self.device.type == 'cuda':
            print(f"  Model device: {next(self.model.parameters()).device}")
            print(f"  CUDA available: {torch.cuda.is_available()}")
            print(f"  CUDA device count: {torch.cuda.device_count()}")
            print(f"  CUDA device name: {torch.cuda.get_device_name(0)}")
        
        self.preprocess = weights.transforms()

    def get_vector(self, folder_path, required_frames, config: 'Config' = None):
        # Use provided config or create default valid_exts
        valid_exts = ('.jpg', '.jpeg', '.png', '.bmp') if config is None else config.valid_exts
        files = sorted([f for f in os.listdir(folder_path)
                        if os.path.isfile(os.path.join(folder_path, f)) and f.lower().endswith(valid_exts)])
        
        if len(files) < required_frames:
            return None 

        tensors = []
        for img_name in files[:required_frames]:
            img_path = os.path.join(folder_path, img_name)
            try:
                img = Image.open(img_path).convert('RGB')
                tensors.append(self.preprocess(img))
            except KeyboardInterrupt:
                raise  # Re-raise KeyboardInterrupt to allow stopping
            except Exception as e:
                print(f"\n[WARN] Failed to read image: {img_path} -> {type(e).__name__}: {e}")
                continue  # Skip bad image and continue
        
        # If we don't have enough valid tensors, return None
        if len(tensors) < required_frames:
            return None

        input_batch = torch.stack(tensors).to(self.device)
        
        # Debug: verify batch is on device
        if self.device.type == 'cuda':
            if torch.cuda.is_available():
                mem_before = torch.cuda.memory_allocated()
        
        with torch.no_grad():
            features = self.model(input_batch)
        
        # Ensure GPU computation finishes before continuing
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
            if torch.cuda.is_available():
                mem_after = torch.cuda.memory_allocated()
            
        return features.cpu().numpy()


class VectorGenerator:
    def __init__(self, config: Config):
        self.cfg = config
        self.extractor = FeatureExtractor()
        self.raw_data = [] # Stores
        self.labels = []

    def run(self):
        print(f"Scanning data roots: {self.cfg.data_roots}")
        print(f"Using device: {self.extractor.device}")
        if self.extractor.device.type == 'cuda':
            print(f"GPU Memory available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            initial_mem = torch.cuda.memory_allocated()
            print(f"Initial GPU memory allocated: {initial_mem / 1e6:.2f} MB")

        for root in self.cfg.data_roots:
            root_path = Path(root)

            if not root_path.is_dir():
                print(f"[WARN] Data root not found or not a directory: {root_path}")
                continue

            raw_data_root = []
            labels_root = []

            # Kiểm tra xem có các label folder trực tiếp không
            subfolders_at_root = [d for d in os.listdir(root_path) if (root_path / d).is_dir()]
            has_direct_labels = any(label.lower() in self.cfg.labels for label in subfolders_at_root)

            print(f"-> Processing root: {root}")
            print(f"   Direct labels found: {has_direct_labels}")

            if has_direct_labels:
                # Cấu trúc: root -> label -> set (không có person folder)
                for label in sorted(os.listdir(root_path)):
                    label_key = label.lower()
                    if label_key not in self.cfg.labels:
                        continue
                    
                    l_path = root_path / label
                    if not l_path.is_dir():
                        continue
                    
                    class_idx = self.cfg.labels[label_key]
                    sets = sorted([s for s in os.listdir(l_path) if (l_path / s).is_dir()])
                    
                    print(f"   Label '{label}': {len(sets)} sets found")

                    for set_folder in tqdm(sets, desc=f"Label: {label}", unit="set"):
                        s_path = l_path / set_folder

                        # Extract vector
                        vector_3d = self.extractor.get_vector(s_path, self.cfg.batch_size, self.cfg)

                        if vector_3d is not None:
                            raw_data_root.append(vector_3d)
                            labels_root.append(class_idx)
            else:
                # Cấu trúc: root -> person -> label -> set
                persons = sorted([p for p in os.listdir(root_path) if (root_path / p).is_dir()])

                if not persons:
                    print(f"[WARN] No persons/labels found in data root: {root_path}")
                    continue

                print(f"   Persons found: {len(persons)}")

                for person in tqdm(persons, desc=f"Persons", unit="person"):
                    p_path = root_path / person

                    for label in sorted(os.listdir(p_path)):
                        label_key = label.lower()
                        if label_key not in self.cfg.labels:
                            continue
                        l_path = p_path / label
                        if not l_path.is_dir():
                            continue
                        class_idx = self.cfg.labels[label_key]

                        sets = sorted([s for s in os.listdir(l_path) if (l_path / s).is_dir()])

                        for set_folder in tqdm(sets, desc=f"{person}/{label}", unit="set", leave=False):
                            s_path = l_path / set_folder

                            # Extract vector
                            vector_3d = self.extractor.get_vector(s_path, self.cfg.batch_size, self.cfg)

                            if vector_3d is not None:
                                raw_data_root.append(vector_3d)
                                labels_root.append(class_idx)

            suffix = root_path.name or "root"
            self.save_files_for(suffix, raw_data_root, labels_root)
            
            # Accumulate into global lists
            if raw_data_root:
                self.raw_data.extend(raw_data_root)
                self.labels.extend(labels_root)

        # After processing all data roots, save combined master files
        self.save_files()

    def save_files(self):
        if self.raw_data:
            X_seq = np.array(self.raw_data)
            y = np.array(self.labels)

            print("\n" + "="*30)
            print("GENERATING OUTPUT VECTORS (combined)")

            print(f"1. Saving LSTM Vectors: {X_seq.shape}...")
            np.save(self.cfg.out_lstm, X_seq)

            print("   Calculating Statistics (Mean + Std) for RF...")
            mean_vec = np.mean(X_seq, axis=1)
            std_vec  = np.std(X_seq, axis=1)
            X_flat   = np.concatenate([mean_vec, std_vec], axis=1)

            print(f"2. Saving RF Vectors:   {X_flat.shape}...")
            np.save(self.cfg.out_rf, X_flat)

            np.save(self.cfg.out_labels, y)
            print("="*30)
            print("Done. Ready for training.")

    def save_files_for(self, suffix: str, raw_data: list, labels: list):
        """Save outputs for a single data root using a suffix derived from the root name."""
        if not raw_data:
            print(f"No data found for root '{suffix}'. Skipping save.")
            return

        safe_suffix = suffix.replace(os.sep, '_')

        X_seq = np.array(raw_data)
        y = np.array(labels)

        print("\n" + "="*30)
        print(f"GENERATING OUTPUT VECTORS for: {safe_suffix}")

        out_lstm = os.path.splitext(self.cfg.out_lstm)[0] + f"_{safe_suffix}.npy"
        out_rf   = os.path.splitext(self.cfg.out_rf)[0] + f"_{safe_suffix}.npy"
        out_labels = os.path.splitext(self.cfg.out_labels)[0] + f"_{safe_suffix}.npy"

        print(f"1. Saving LSTM Vectors: {X_seq.shape} -> {out_lstm}")
        np.save(out_lstm, X_seq)

        print("   Calculating Statistics (Mean + Std) for RF...")
        mean_vec = np.mean(X_seq, axis=1)
        std_vec  = np.std(X_seq, axis=1)
        X_flat   = np.concatenate([mean_vec, std_vec], axis=1)

        print(f"2. Saving RF Vectors:   {X_flat.shape} -> {out_rf}")
        np.save(out_rf, X_flat)

        np.save(out_labels, y)
        print("="*30)
        print(f"Done saving outputs for: {safe_suffix}")


if __name__ == "__main__":
    cfg = Config()
    app = VectorGenerator(cfg)
    app.run()
