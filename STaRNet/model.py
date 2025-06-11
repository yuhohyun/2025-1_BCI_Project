import torch
import torch.nn as nn


class STaRNet(nn.Module):
    def __init__(self, 
                 num_channels=22,         # EEG 채널 수
                 time_length=1000,       # 시간 길이
                 fusion_channels=50,       # 공간 fusion 출력 채널 수 (fusion 후)
                 temporal_channels=150,       # temporal 블록의 출력 채널 수 (각 스케일 당); depthwise conv 조건상 temporal_channels는 fusion_channels의 배수여야 함.
                 temporal_kernel_sizes=[64, 32, 16],  # 다중 스케일 temporal 커널 크기
                 spatial_kernel_sizes=[1, 3, 5], # 다중 스케일 spatial 커널 높이
                 spatial_expansion=32, # spatial expansion coefficient (S_e)
                 mapped_dim=64, # bilinear mapping 후 저차원 SPD 행렬의 차원
                 num_classes=4):
        super(STaRNet, self).__init__()
        
        self.num_channels = num_channels
        self.time_length = time_length
        self.fusion_channels = fusion_channels
        # depthwise 컨볼루션을 위해 temporal_channels가 fusion_channels의 배수가 되도록 강제
        if temporal_channels % fusion_channels != 0:
            temporal_channels = fusion_channels
        self.temporal_channels = temporal_channels
        self.temporal_kernel_sizes = temporal_kernel_sizes
        self.spatial_kernel_sizes = spatial_kernel_sizes
        self.mapped_dim = mapped_dim
        self.spatial_expansion = spatial_expansion

        # 1. 다중 스케일 공간 특징 추출
        # 각 spatial 스케일마다: 커널 크기 (h, 1)와 출력 채널 수 d_i (d_i = max(1, spatial_expansion // (num_channels - h + 1)))
        self.spatial_branches = nn.ModuleList()
        self.spatial_channel_dims = []  # 각 branch의 출력 채널 수 × 출력 높이(S_i)를 저장
        for kernel_size in spatial_kernel_sizes:
            output_height = num_channels - kernel_size + 1  # 출력 높이
            output_channels = max(1, spatial_expansion // output_height)
            self.spatial_channel_dims.append(output_channels * output_height)
            self.spatial_branches.append(nn.Sequential(
                nn.Conv2d(1, output_channels, kernel_size=(kernel_size, 1), bias=False),
                nn.BatchNorm2d(output_channels)
            ))
        # 전체 spatial 분기의 출력 높이
        self.total_spatial_channels = sum(self.spatial_channel_dims)
        # 다중 스케일 spatial 특징을 융합하는 convolution: 커널 크기 (total_spatial_channels, 1)
        self.spatial_fuse = nn.Sequential(
            nn.Conv2d(1, fusion_channels, kernel_size=(self.total_spatial_channels, 1), bias=False),
            nn.BatchNorm2d(fusion_channels)
        )
        
        # 2. 다중 스케일 Temporal 특징 추출 (depthwise convolution)
        # 각 temporal 블록은 입력 채널 fusion_channels를 받아 출력 채널 temporal_channels를 생성 (groups = fusion_channels)
        self.temporal_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(fusion_channels, temporal_channels, kernel_size=(1, k), padding=(0, k // 2), groups=fusion_channels, bias=False),
                nn.BatchNorm2d(temporal_channels)
            )
            for k in temporal_kernel_sizes
        ])
        # 최종 temporal 특징 차원: temporal_channels * (temporal 스케일 수)
        self.output_dim = temporal_channels * len(temporal_kernel_sizes)

        # 3. Bilinear mapping
        # high-dimensional covariance 행렬에서 저차원 SPD 행렬로 투영
        self.bilinear_map = nn.Parameter(torch.randn(self.output_dim, mapped_dim))

        self._initialize_bilinear() 

        # 4. 접공간 특징을 벡터화한 후 분류하기 위한 Fully Connected layer
        self.fc = nn.Linear((mapped_dim * (mapped_dim + 1)) // 2, num_classes)

    def _initialize_bilinear(self):
        # semi-orthonormal initialization via QR
        input_dim, output_dim = self.bilinear_map.shape
        raw_weights = torch.randn(input_dim, output_dim, device=self.bilinear_map.device) * 0.01
        Q, _ = torch.linalg.qr(raw_weights)          # Q: (D×D)
        with torch.no_grad():
            self.bilinear_map.copy_(Q[:, :output_dim])
    
    def forward(self, x):
        # 입력 x shape: (B, 1, C, T)
        batch_size = x.size(0)
        # 1. 다중 스케일 공간 특징 추출
        spatial_features = []
        # 각 branch: (B, d_i, S_i, T) -> reshape -> (B, 1, d_i*S_i, T)
        for branch in self.spatial_branches:
            out = branch(x)
            out = out.view(batch_size, 1, -1, self.time_length)
            spatial_features.append(out)
        # 모든 스케일의 결과를 concat: (B, 1, total_spatial_channels, T)
        x_spatial = torch.cat(spatial_features, dim=2)
        # spatial fusion: convolution으로 융합하여 (B, fusion_channels, 1, T) 생성
        x_fused = self.spatial_fuse(x_spatial)  # (B, fusion_channels, 1, T)
        
        # 2. 다중 스케일 temporal 특징 추출
        temporal_features = []
        for block in self.temporal_blocks:
            temporal_features.append(block(x_fused))  # 각 결과: (B, temporal_channels, 1, T)
        # concat하여 (B, temporal_channels*num_scales, 1, T) -> squeeze하여 (B, temporal_channels*num_scales, T)
        h = torch.cat(temporal_features, dim=1)
        h = h.squeeze(2)
        
        # 3. 공분산 행렬 계산 및 리만 다양체 임베딩
        # 공분산 계산: (B, d, d) with d = temporal_channels*num_scales
        h_t = h.transpose(1, 2)
        cov = torch.bmm(h, h_t) / (self.time_length - 1)

         # ---- shrinkage: Σ' = (1-α)Σ + α μ I ----
        alpha = 0.05
        dim = cov.size(1)
        mu = cov.diagonal(0,1,2).mean(dim=1)          # (B,)
        mu = mu.view(batch_size,1,1).expand(batch_size,dim,dim)
        cov = (1-alpha)*cov + alpha * mu * torch.eye(dim,device=cov.device).unsqueeze(0)

        # bilinear mapping: Wᵀ Σ W
        W = self.bilinear_map     # (D, d_bm)
        tmp = cov @ W             # (B, D, d_bm)
        mapped_cov = W.transpose(0,1) @ tmp  # (B, d_bm, d_bm)

        # log‐euclidean mapping
        eigvals, eigvecs = torch.linalg.eigh(mapped_cov)
        log_eig = torch.log(torch.clamp(eigvals, min=1e-6))
        log_mapped = eigvecs @ torch.diag_embed(log_eig) @ eigvecs.transpose(-1,-2)

        # upper‐triangular vectorization
        idx = torch.triu_indices(self.mapped_dim, self.mapped_dim)
        features = log_mapped[:, idx[0], idx[1]]  # (B, d_bm*(d_bm+1)//2)

        return self.fc(features)


