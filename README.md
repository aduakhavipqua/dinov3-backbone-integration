# Melody_AI – CoTMix + DINOv3

Dự án thử nghiệm thích nghi miền (domain adaptation) cho bài toán nhận dạng hoạt động của con người (HAR) từ cảm biến quán tính, dựa trên **CoTMix** (Contrastive Temporal Mixup) và thay bộ mã hóa 1D-CNN gốc bằng **DINOv3 ViT-S/16**. Kèm theo là một ứng dụng demo **Gradio** (giao diện Melody_AI) có thể đóng gói bằng Docker.

> **Lưu ý:** DINOv3 là mô hình thị giác. Việc đưa tín hiệu cảm biến vào DINOv3 qua ảnh GAF là một **thiết kế thử nghiệm**, không thuộc bài báo CoTMix gốc. Hãy xem đây là một ablation và so sánh với CNN gốc trong cùng điều kiện.

---

## Mục lục

1. [Tính năng](#1-tính-năng)
2. [Cấu trúc thư mục](#2-cấu-trúc-thư-mục)
3. [Cài đặt](#3-cài-đặt)
4. [Chuẩn bị dữ liệu UCI-HAR](#4-chuẩn-bị-dữ-liệu-uci-har)
5. [Huấn luyện](#5-huấn-luyện)
6. [Kết quả và file đầu ra](#6-kết-quả-và-file-đầu-ra)
7. [Chạy ứng dụng demo](#7-chạy-ứng-dụng-demo)
8. [Docker](#8-docker)
9. [Cách hoạt động](#9-cách-hoạt-động)
10. [Hạn chế và lưu ý](#10-hạn-chế-và-lưu-ý)

---

## 1. Tính năng

- Giữ nguyên backbone `CNN` của CoTMix làm baseline.
- Thêm backbone `DINOv3ViTS16`: tín hiệu `[B, 9, 128]` → ảnh GAF 3 kênh → resize 224×224 → DINOv3 → embedding CLS (384 chiều) → bộ phân loại.
- Temporal mixup của CoTMix vẫn thực hiện trên tensor chuỗi thời gian, trước khi chuyển sang ảnh.
- DINOv3 **đóng băng mặc định**; dùng `--unfreeze_dinov3` để fine-tune với learning rate riêng.
- Script chuyển UCI-HAR thô sang định dạng `.pt` mà CoTMix yêu cầu.
- Demo Gradio: nhập/tải tín hiệu 9 kênh × 128 mẫu, xem biểu đồ và xác suất 6 lớp hoạt động.

## 2. Cấu trúc thư mục

```text
.
├── main.py                      # Điểm vào huấn luyện (argparse)
├── trainer.py                   # cross_domain_trainer: huấn luyện, đánh giá, ghi log
├── algorithms.py                # Thuật toán CoTMix (mixup + các loss)
├── dataloader.py                # Load_Dataset, data_generator
├── utils.py                     # Seed, logging, checkpoint, metric, DEV risk
├── prepare_uci_har_cotmix.py    # Chuyển UCI-HAR thô -> train_<id>.pt / test_<id>.pt
├── app.py                       # Demo Gradio (Melody_AI)
├── Dockerfile
├── requirements.txt             # Phụ thuộc cho demo / Docker
├── requirements-dinov3.txt      # Phụ thuộc thêm cho DINOv3
├── DINOv3_UCI_HAR.md            # Ghi chú kỹ thuật về phần tích hợp DINOv3
├── cotmix_uci_har.pt            # state_dict sau huấn luyện
├── cotmix_uci_har.pkl           # toàn bộ đối tượng algorithm (torch.save)
├── configs/                     # data_model_configs.py, hparams.py, sweep_params.py
└── models/                      # models.py, loss.py, (dinov3.py)
```

> Hai thư mục `configs/` và `models/` thuộc mã nguồn CoTMix gốc và được `trainer.py`, `algorithms.py` import. Hãy bảo đảm chúng có mặt trong repository.

## 3. Cài đặt

Yêu cầu: Python 3.10+, khuyến nghị có GPU CUDA để huấn luyện.

```bash
# 1. Phụ thuộc cốt lõi của CoTMix (torch, torchvision, sklearn, pandas, wandb, ...)
pip install torch torchvision numpy pandas scikit-learn wandb matplotlib openpyxl skorch

# 2. Phụ thuộc cho DINOv3
pip install -r requirements-dinov3.txt
```

DINOv3 được tải từ Hugging Face (`facebook/dinov3-vits16-pretrain-lvd1689m`). Bạn có thể cần chấp nhận điều khoản truy cập mô hình và đăng nhập:

```bash
huggingface-cli login
```

## 4. Chuẩn bị dữ liệu UCI-HAR

CoTMix không đọc file text thô của UCI-HAR mà đọc file PyTorch đã xử lý:

```text
data/
└── HAR/
    ├── train_<subject>.pt
    └── test_<subject>.pt
```

Mỗi file là một dict gồm:

| Khóa      | Kiểu / shape                | Ý nghĩa                       |
|-----------|-----------------------------|-------------------------------|
| `samples` | `float32`, `[N, 9, 128]`    | 9 kênh cảm biến, 128 bước     |
| `labels`  | `int64`, `[N]`              | Nhãn 0–5                      |

**Bước 1 – Tải UCI-HAR Dataset** và giải nén sao cho có `train/Inertial Signals/`, `test/Inertial Signals/`, `subject_*.txt`, `y_*.txt`.

**Bước 2 – Sửa đường dẫn** ở đầu `prepare_uci_har_cotmix.py`:

```python
DATASET_ROOT = Path(r"đường/dẫn/tới/UCI-HAR Dataset")
OUTPUT_DIR   = Path(r"đường/dẫn/tới/data/HAR")
TEST_SIZE    = 0.2
RANDOM_SEED  = 42
SUBJECTS     = "all"   # hoặc [2, 6, 11]
```

**Bước 3 – Chạy:**

```bash
python prepare_uci_har_cotmix.py
```

Script sẽ: đọc 9 tín hiệu quán tính (`body_acc`, `body_gyro`, `total_acc` theo x/y/z), gộp tập train/test gốc, đổi nhãn `1..6 → 0..5`, rồi **chia lại train/test theo từng subject** (stratified khi có thể) và lưu thành `train_<id>.pt`, `test_<id>.pt`.

Các subject id phải khớp danh sách `scenarios` của HAR trong `configs/data_model_configs.py`.

Nhãn hoạt động:

| ID | Hoạt động       |
|----|-----------------|
| 0  | Đi bộ           |
| 1  | Lên cầu thang   |
| 2  | Xuống cầu thang |
| 3  | Ngồi            |
| 4  | Đứng            |
| 5  | Nằm             |

## 5. Huấn luyện

Chạy thử nhanh (GPU, 1 run, 2 epoch):

```bash
python main.py --dataset HAR --backbone DINOv3ViTS16 --data_path ./data/ \
  --device cuda:0 --num_runs 1 --batch_size 16 --epochs 2
```

Chạy thử trên CPU (rất chậm):

```bash
python main.py --dataset HAR --backbone DINOv3ViTS16 --data_path ./data/ \
  --device cpu --num_runs 1 --batch_size 2 --epochs 1
```

Fine-tune DINOv3:

```bash
python main.py --dataset HAR --backbone DINOv3ViTS16 --data_path ./data/ \
  --device cuda:0 --num_runs 1 --batch_size 8 --epochs 20 \
  --unfreeze_dinov3 --dinov3_lr 0.00001 --classifier_lr 0.001
```

Baseline CNN gốc:

```bash
python main.py --dataset HAR --backbone CNN --data_path ./data/ --device cuda:0
```

### Tham số chính

| Tham số | Mặc định | Mô tả |
|---|---|---|
| `--backbone` | `DINOv3ViTS16` | `CNN` hoặc `DINOv3ViTS16` |
| `--dataset` | `HAR` | Bộ dữ liệu (WISDM, EEG, HAR, HHAR) |
| `--da_method` | `CoTMix` | Thuật toán thích nghi miền |
| `--data_path` | `./data/` | Thư mục chứa `HAR/` |
| `--device` | `cuda:0` | `cpu` hoặc `cuda:N` |
| `--num_runs` | `3` | Số lần chạy với các seed khác nhau |
| `--batch_size` | `16` | Chỉ áp dụng cho DINOv3 |
| `--epochs` | `50` | Chỉ áp dụng cho DINOv3 |
| `--dinov3_model_name` | `facebook/dinov3-vits16-pretrain-lvd1689m` | Model ID trên Hugging Face |
| `--dinov3_image_size` | `224` | Kích thước ảnh đầu vào |
| `--unfreeze_dinov3` | tắt | Fine-tune DINOv3 |
| `--dinov3_lr` | `1e-5` | LR của DINOv3 khi fine-tune |
| `--classifier_lr` | `1e-3` | LR của bộ phân loại |
| `--is_sweep` | tắt | Chạy W&B hyperparameter sweep |

Với backbone `CNN`, `batch_size` và số epoch lấy từ `configs/hparams.py` như CoTMix gốc. Optimizer: `AdamW` với hai nhóm learning rate cho DINOv3, `Adam` cho CNN.

W&B chạy ở chế độ **offline** khi không sweep (đổi `mode="offline"` trong `trainer.py` nếu muốn đồng bộ online).

## 6. Kết quả và file đầu ra

Sau mỗi lần chạy, kết quả nằm trong `experiments_logs/<experiment_description>/<run_description>/`:

- `<src>_to_<trg>_run_<k>/`: log, `checkpoint.pt`, `classification_report.xlsx`, `scores.xlsx`
- `Average_results.xlsx`, `std_results.xlsx`: Accuracy và macro-F1 trung bình / độ lệch chuẩn theo scenario
- `MODEL_BACKUP_FILES/`: bản sao mã nguồn tại thời điểm chạy

Ngoài ra:

- `trainer.py` ghi `cotmix_uci_har.pt` (**state_dict**) ở thư mục gốc.
- `main.py` ghi `cotmix_uci_har.pkl` (**toàn bộ đối tượng** `algorithm`, bằng `torch.save`).

Cả hai file chỉ phản ánh scenario/run **cuối cùng** của lần chạy.

## 7. Chạy ứng dụng demo

```bash
pip install gradio numpy pandas scikit-learn matplotlib torch
python app.py
```

Mở `http://localhost:7860` (đổi cổng bằng biến môi trường `PORT`).

Cách dùng:

1. Chọn một **mẫu có sẵn** (Ngồi, Đứng, Đi bộ, Nằm, Lên cầu thang) hoặc dán/tải tín hiệu `.csv`, `.txt`, `.json`.
2. Bấm dự đoán để xem biểu đồ 9 kênh, hoạt động dự đoán, độ tin cậy và lịch sử gần đây.

Định dạng đầu vào mong đợi: 9 kênh × 128 bước thời gian.

## 8. Docker

```bash
docker build -t melody_ai .
docker run --rm -p 7860:7860 melody_ai
```

Image dùng `python:3.10-slim`, cài PyTorch bản CPU, chạy bằng người dùng không phải root và mở cổng `7860`, phù hợp để triển khai lên Hugging Face Spaces (Docker SDK).

## 9. Cách hoạt động

```text
src_x, trg_x  [B, 9, 128]
   │
   ├─ Temporal mixup (CoTMix) ──► src_dominant, trg_dominant
   │
   ▼
Feature extractor
   ├─ CNN                      : 1D-CNN trên chuỗi thời gian
   └─ DINOv3ViTS16             : GAF (3 kênh) → 224×224 → ViT-S/16 → CLS (384)
   ▼
Classifier ──► logits
```

Hàm mất mát tổng hợp: cross-entropy trên nguồn + conditional entropy trên đích + supervised contrastive (nguồn) + NT-Xent (đích), với trọng số lấy từ `hparams`.

GAF-to-RGB trong DINOv3: với UCI-HAR, 9 kênh được chia thành 3 nhóm × 3 kênh, trung bình mỗi nhóm thành một mặt phẳng ảnh. Chi tiết xem thêm trong [`DINOv3_UCI_HAR.md`](DINOv3_UCI_HAR.md).

## 10. Hạn chế và lưu ý

- **Demo chưa chắc dùng mô hình thật.** `app.py` nạp mô hình bằng `torch.jit.load`, trong khi `cotmix_uci_har.pt` do trainer xuất là `state_dict`. Khi nạp thất bại, app **chuyển sang xác suất giả lập theo mẫu đã chọn**. Để demo dùng suy luận thật, cần xuất mô hình dạng TorchScript hoặc sửa `app.py` để dựng lại kiến trúc rồi `load_state_dict`.
- **Thống kê trên giao diện chỉ là minh họa.** Lịch sử ban đầu, tổng số mẫu (1248), thời gian xử lý (`12.4s`) và số lớp trong thẻ thống kê là giá trị cố định trong `app.py`.
- **Phụ thuộc Docker chưa đủ cho mô hình thật.** `requirements.txt` không có `torchvision`, `transformers` hay `skorch`; nếu muốn chạy backbone DINOv3 trong container, hãy bổ sung.
- **`trainer.py` còn giá trị cấu hình của tác giả gốc**: `entity='emadeldeen'` trong `wandb.sweep` cần đổi nếu bạn chạy sweep.
- **Cách chia dữ liệu:** `prepare_uci_har_cotmix.py` chia lại train/test ngẫu nhiên trong từng subject, nên khác với giao thức chia theo subject của UCI-HAR gốc. Không so sánh trực tiếp với số liệu công bố nếu không nêu rõ điều này.
- **Đường dẫn trong `prepare_uci_har_cotmix.py`** đang là đường dẫn Windows cá nhân, cần sửa trước khi chạy.
- **`.gitattributes`** đẩy `*.pt`, `*.pkl`, `*.txt` qua Git LFS; cần cài `git lfs` trước khi clone/push.
- Để so sánh công bằng với CNN, dùng cùng scenario, seed và ghi lại: adapter GAF, checkpoint DINOv3, trạng thái đóng băng/fine-tune và phiên bản thư viện.

## Tham khảo

- CoTMix: Eldele et al., *Contrastive Domain Adaptation for Time-Series via Temporal Mixup*.
- UCI HAR Dataset: Anguita et al., *A Public Domain Dataset for Human Activity Recognition Using Smartphones*.
- DINOv3: Meta AI, mô hình `facebook/dinov3-vits16-pretrain-lvd1689m` trên Hugging Face.
