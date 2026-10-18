"""
DINOv3 backbone cho CoTMix (thay cho CNN 1D gốc của AdaTime).

Luồng xử lý:
    x (B, C, L)  --STFT-->  log-spectrogram (B, C, F, T)
                 --1x1 conv (C -> 3)--> (B, 3, F, T)
                 --resize--> (B, 3, S, S)           S = dinov3_img_size (mặc định 224)
                 --DINOv3 ViT--> CLS embedding (B, D)
                 --Linear--> (B, final_out_channels * features_len)

Đầu ra được giữ ở dạng vector phẳng có đúng số chiều
`configs.final_out_channels * configs.features_len`, giống CNN gốc, nên
`classifier(configs)` và `calc_dev_risk` trong utils.py dùng được mà không phải sửa.

Các tuỳ chọn đọc từ `configs` (được trainer gán từ tham số dòng lệnh):
    dinov3_model      : tên model trên Hugging Face
    dinov3_finetune   : False -> đóng băng DINOv3, chỉ train bộ chuyển kênh + projection
    dinov3_img_size   : kích thước ảnh đưa vào ViT (chia hết cho 16)

Lưu ý: weights DINOv3 là gated trên Hugging Face. Cần đồng ý điều khoản trên trang model
và đăng nhập bằng `huggingface-cli login` trước khi chạy.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DINOv3(nn.Module):
    def __init__(self, configs):
        super(DINOv3, self).__init__()
        from transformers import AutoModel  # import tại đây để CNN vẫn chạy khi chưa cài transformers

        model_name = getattr(configs, "dinov3_model", "facebook/dinov3-vits16-pretrain-lvd1689m")
        self.finetune = getattr(configs, "dinov3_finetune", False)
        self.img_size = getattr(configs, "dinov3_img_size", 224)

        in_channels = configs.input_channels
        out_dim = configs.final_out_channels * configs.features_len

        # (1) Bộ chuyển kênh: C kênh cảm biến -> 3 kênh ảnh (có học được)
        self.channel_adapter = nn.Conv2d(in_channels, 3, kernel_size=1)

        # (2) DINOv3
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        if not self.finetune:
            for p in self.encoder.parameters():
                p.requires_grad = False
            self.encoder.eval()

        # (3) Projection về đúng số chiều mà classifier / dev_risk mong đợi
        self.proj = nn.Sequential(nn.Linear(hidden, out_dim), nn.ReLU(), nn.Dropout(0.1))

        # chuẩn hoá theo ImageNet (DINOv3 được tiền huấn luyện với thống kê này)
        self.register_buffer("img_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("img_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def train(self, mode=True):
        super().train(mode)
        if not self.finetune:  # giữ encoder đóng băng ở eval (tắt dropout/norm update)
            self.encoder.eval()
        return self

    def _to_spectrogram(self, x):
        """(B, C, L) -> (B, C, F, T) log-magnitude STFT."""
        B, C, L = x.shape
        n_fft = min(64, L)
        hop = max(1, n_fft // 4)
        window = torch.hann_window(n_fft, device=x.device)
        spec = torch.stft(x.reshape(B * C, L), n_fft=n_fft, hop_length=hop,
                          window=window, return_complex=True)
        spec = torch.log1p(spec.abs())
        return spec.reshape(B, C, spec.shape[-2], spec.shape[-1])

    def _encode(self, img):
        out = self.encoder(pixel_values=img)
        feat = getattr(out, "pooler_output", None)
        if feat is None:
            feat = out.last_hidden_state[:, 0]  # token CLS
        return feat

    def forward(self, x):
        spec = self._to_spectrogram(x)
        img = self.channel_adapter(spec)
        img = F.interpolate(img, size=(self.img_size, self.img_size),
                            mode="bilinear", align_corners=False)

        # chuẩn hoá từng mẫu về thang ~[0,1] rồi áp thống kê ImageNet
        mn = img.amin(dim=(1, 2, 3), keepdim=True)
        mx = img.amax(dim=(1, 2, 3), keepdim=True)
        img = (img - mn) / (mx - mn + 1e-6)
        img = (img - self.img_mean) / self.img_std

        if self.finetune:
            feat = self._encode(img)
        else:
            with torch.no_grad():
                feat = self._encode(img)
        return self.proj(feat)