# REVIEW.md - Tổng Quan Dự Án River Raid RL

> **Mục tiêu**: Đào tạo Agent chơi Atari River Raid bằng thuật toán Rainbow DQN, chạy được trên laptop CPU thấp cấp.

---

## 1. Cấu Trúc Thư Mục

```
river-raid-rl-base/
├── pyproject.toml              # Metadata build & dependencies (Python >=3.10)
├── requirements.txt            # Danh sách pip dependencies
├── README.md                   # Tài liệu chính (LaTeX, benchmark, hướng dẫn)
├── ABLATION.md                 # Hướng dẫn Ablation Study cho 7 thành phần Rainbow
├── CONTRIBUTING.md             # Hướng dẫn đóng góp
├── CITATION.cff                # Metadata trích dẫn học thuật
├── LICENSE                     # Giấy phép MIT
├── mkdocs.yml                  # Cấu hình tài liệu MkDocs
├── .ruff.toml                  # Cấu hình linting (ruff)
├── .gitignore / .gitattributes
│
├── configs/                    # File cấu hình YAML cho huấn luyện
│   └── rainbow.yaml            # Cấu hình Rainbow DQN đầy đủ
│
├── riverraid_rl/               # Thư viện cốt lõi
│   ├── __init__.py             # Đăng ký ALE envs
│   ├── __main__.py
│   ├── config.py               # Dataclass cấu hình (EnvConfig, DQNConfig, RainbowConfig, TrainingConfig)
│   ├── env.py                  # Atari environment wrappers
│   ├── env_hierarchical.py     # Env wrapper theo dõi nhiên liệu + bộ điều khiển phân cấp
│   ├── curriculum.py
│   ├── train.py
│   ├── pbt.py                  # Population-Based Training
│   ├── run_demo.py
│   │
│   ├── agents/                 # Các loại Agent
│   │   ├── base.py             # BaseAgent (abstract class)
│   │   ├── rainbow.py          # RainbowAgent - agent chính
│   │   ├── dqn.py              # DQNAgent - DQN cơ bản
│   │   ├── better_than_human.py # BetterThanHumanRainbowAgent - nâng cao hơn
│   │   ├── hierarchical.py     # HierarchicalRiverRaidAgent - phân cấp
│   │   ├── icm_rainbow.py      # ICMRainbowAgent - curiosity-driven
│   │   ├── random_agent.py     # RandomAgent -_baseline ngẫu nhiên
│   │   └── rule_based.py       # RuleBasedAgent - heuristic
│   │
│   ├── models/                 # Mô hình mạng nơ-ron
│   │   ├── cnn.py              # DQNCNN, DuelingDQN, CategoricalDuelingDQN, CategoricalDuelingDQNAttention
│   │   ├── noisy.py            # NoisyLinear, NoisyDuelingDQN (NoisyNets)
│   │   ├── attention.py        # SpatialAttention, ChannelAttention, AttentionDQN, AuxiliaryDQN
│   │   └── icm.py              # IntrinsicCuriosityModule (ICM)
│   │
│   ├── memory/                 # Replay Buffer
│   │   └── replay.py           # ReplayBuffer, PrioritizedReplayBuffer, NStepReplayBuffer, NStepPrioritizedReplayBuffer
│   │
│   ├── utils/                  # Tiện ích
│   │   ├── evaluation.py       # Hàm evaluate() đánh giá agent
│   │   ├── logger.py           # Logger ghi metrics ra JSON
│   │   └── progress.py         # ProgressTracker theo dõi tiến độ vs benchmark
│   │
│   └── scripts/                # Script nội bộ
│       ├── train_demo.py
│       ├── train_demo2.py
│       ├── eval_all.py
│       ├── final_results.py
│       └── speed_test.py
│
├── experiments/                # Script huấn luyện & đánh giá
│   ├── train.py                # Script huấn luyện chính (YAML-driven)
│   ├── evaluate.py             # Đánh giá checkpoint
│   ├── train_riverraid.py      # Script huấn luyện gốc (Rainbow)
│   ├── train_cpu.py            # Huấn luyện tối ưu CPU
│   ├── train_improved.py       # Rainbow nâng cao (no reward clip, n-step=3)
│   ├── train_optimized.py      # Tối ưu GPU + SyncVectorEnv
│   └── train_human_level.py    # Huấn luyện mục tiêu level người (AsyncVectorEnv + torch.compile)
│
├── scripts/                    # Script tiện ích (không cần training)
│   ├── report_existing.py      # Tổng hợp kết quả từ progress.json
│   ├── instantiate_agents.py   # Kiểm tra mọi agent có thể khởi tạo
│   ├── render_gameplay.py      # Render video gameplay
│   ├── eval_checkpoints.py     # Đánh giá nhiều checkpoints
│   ├── eval_dqn50k.py          # Đánh giá DQN 50K
│   ├── _eval_baselines.py      # Đánh giá baseline agents
│   └── benchmark_speed.py      # Benchmark tốc độ training
│
├── results/                    # Phân tích & trực quan hóa
│   ├── plot_training.py        # Vẽ đồ thị learning curves
│   └── ablation_table.py       # Bảng so sánh Ablation Study
│
├── tests/                      # Unit tests (pytest)
│   ├── test_agents_instantiate.py  # Test khởi tạo mọi agent
│   ├── test_config.py          # Test parse YAML config
│   ├── test_instantiate_exported.py
│   ├── test_rainbow_low_compute.py
│   └── test_better_than_human_agent.py
│
├── notebooks/                  # Jupyter Notebook
│   └── quickstart.ipynb        # Hướng dẫn bắt đầu nhanh
│
├── docs/                       # Tài liệu MkDocs
│   ├── index.md
│   ├── PLAN_HUMAN_LEVEL.md
│   └── RIVERRAID_RL_REPORT.md
│
├── media/                      # Video demo
├── checkpoints/                # Lưu checkpoints mô hình
├── results/                    # Kết quả phân tích
├── runs/                       # Kết quả training runs
├── logs/                       # Logs training
└── .github/workflows/tests.yml # CI pipeline
```

---

## 2. Chi Tiết Nội Dung Từng File

### 2.1. Thư Viện Cốt Lõi (`riverraid_rl/`)

#### `config.py` - Cấu Hình Dataclass
- **`EnvConfig`**: env_id (`ALE/Riverraid-v5`), frame_stack=4, frame_skip=4, screen_size=84, grayscale=True, max_episode_steps=108000
- **`DQNConfig`**: learning_rate=0.00025, batch_size=32, buffer_capacity=100000, gamma=0.99, epsilon_decay=250K, hidden_dim=512
- **`RainbowConfig`** (kế thừa DQNConfig): num_atoms=51, v_min=-10, v_max=10, n_step=3, alpha=0.6, beta_start=0.4
- **`TrainingConfig`**: total_timesteps=10M, eval_freq=250K, seed=42, device="cpu"

#### `env.py` - Environment Wrappers
- **`FireResetEnv`**: Tự động bấm FIRE khi reset
- **`EpisodicLifeEnv`**: Kết thúc episode khi mất 1 mạng (thay vì đợi game over)
- **`MaxAndSkipEnv`**: Lấy max frame trong 4 frame liên tiếp (frame skip=4)
- **`ClipRewardEnv`**: Clamp reward về {-1, 0, +1}
- **`make_riverraid_env()`**: Hàm chính tạo env với đầy đủ wrappers

#### `agents/rainbow.py` - RainbowAgent (179 dòng)
- Triển khai đầy đủ Rainbow DQN: Double DQN + Dueling + PER + N-step + C51
- **`act()`**: Epsilon-greedy action selection
- **`update()`**: Sample từ PER buffer, tính cross-entropy loss trên phân phối C51, cập nhật priorities
- **`_project_distribution()`**: Project target distribution lên atoms
- Lưu: q_network, target_network, optimizer, steps, epsilon, beta

#### `agents/dqn.py` - DQNAgent (108 dòng)
- DQN cơ bản với Double Q-learning, MSE loss
- Dùng `ReplayBuffer` thường (không có PER)

#### `agents/better_than_human.py` - BetterThanHumanRainbowAgent (342 dòng)
- Nâng cấp với NoisyNets (không cần epsilon-greedy), CosineAnnealingLR scheduler
- Hỗ trợ `act_batch()` cho vectorized environments
- Hỗ trợ AMP (Automatic Mixed Precision) trên GPU
- `NoisyCategoricalDuelingDQN`: Mạng Dueling C51 với NoisyLinear layers
- Config: hidden_dim=256, gamma=0.997, n_step=5, v_max=100

#### `agents/hierarchical.py` - HierarchicalRiverRaidAgent (190 dòng)
- 2 mạng: nav_network (di chuyển) + fuel_network (lấy nhiên liệu)
- Meta-network quyết định goal (0=navigate, 1=fuel)
- Tích hợp ICM (Intrinsic Curiosity Module) cho exploration

#### `agents/icm_rainbow.py` - ICMRainbowAgent (191 dòng)
- Rainbow DQN kết hợp ICM cho curiosity-driven exploration
- Reward = external reward + eta * intrinsic_reward

#### `agents/random_agent.py` & `rule_based.py`
- RandomAgent: Chọn hành động ngẫu nhiên (baseline)
- RuleBasedAgent: Heuristic - tìm vị trí máy bay, tìm trục đường, đi theo

#### `models/cnn.py` - Các Mạng CNN (158 dòng)
- **`DQNCNN`**: CNN cơ bản (3 conv layers → FC → Q-values)
- **`DuelingDQN`**: Tách value stream + advantage stream
- **`CategoricalDuelingDQN`**: Dueling + C51 distributional (51 atoms)
- **`CategoricalDuelingDQNAttention`**: Thêm SpatialAttention + ChannelAttention

#### `models/noisy.py` - NoisyNets (70 dòng)
- **`NoisyLinear`**: Layer tuyến tính có noise learned (factorized Gaussian noise)
- **`NoisyDuelingDQN`**: Dueling DQN sử dụng NoisyLinear thay FC layers

#### `models/attention.py` - Attention Mechanisms (126 dòng)
- **`SpatialAttention`**: Attention không gian (conv1x1 → sigmoid → multiply)
- **`ChannelAttention`**: Attention kênh (AdaptiveAvgPool → Conv → Sigmoid)
- **`AttentionDQN`**: CNN + Attention cho hierarchical agent
- **`AuxiliaryDQN`**: Multi-head prediction (policy, value, fuel, enemy_density, position)

#### `models/icm.py` - Intrinsic Curiosity Module (102 dòng)
- **`FeatureEncoder`**: Trích xuất đặc trưng từ frame
- **`ForwardModel`**: Dự đoán feature tiếp theo từ (state, action)
- **`InverseModel`**: Dự đoán action từ (state, next_state)
- Loss = inverse_loss + forward_loss, intrinsic_reward = L2 distance

#### `memory/replay.py` - Replay Buffers (268 dòng)
- **`ReplayBuffer`**: FIFO buffer cơ bản
- **`PrioritizedReplayBuffer`**: Sum-tree-based PER với importance sampling weights
- **`NStepReplayBuffer`**: N-step bootstrapping kết hợp ReplayBuffer
- **`NStepPrioritizedReplayBuffer`**: N-step + PER (dùng cho RainbowAgent)

#### `utils/progress.py` - ProgressTracker (112 dòng)
- Benchmark targets: Theoretical Max=1M, Human Expert=13,513, Rainbow SOTA=20,675, DQN Nature=8,311
- Tính % so với mỗi target, lưu progress.json

#### `utils/evaluation.py` - evaluate() (39 dòng)
- Chạy agent greedily (không exploration) trong N episodes
- Trả về: mean_reward, std_reward, min/max_reward, mean_length

#### `utils/logger.py` - Logger (38 dòng)
- Ghi metrics (loss, q_value, epsilon, eval/mean) ra JSON

### 2.2. Scripts Huấn Luyện (`experiments/`)

#### `train.py` - Script Chính (YAML-driven)
- Đọc config từ YAML → tạo env + agent → training loop → logging → checkpointing
- Hỗ trợ cả Rainbow và DQN

#### `train_riverraid.py` - Script Gốc (233 dòng)
- Rainbow agent, 1M steps mặc định, no reward clipping
- Hỗ trợ: --quick-test, --big-model, --resume, --targets

#### `train_cpu.py` - CPU Training (98 dòng)
- Rainbow với 21 atoms, 50K buffer, 200K steps
- Tối ưu cho CPU thấp cấp

#### `train_improved.py` - Rainbow Nâng Cao (120 dòng)
- 51 atoms, 100K buffer, n-step=3, no reward clipping
- Progress logging chi tiết

#### `train_optimized.py` - GPU + Vectorized (253 dòng)
- SyncVectorEnv với nhiều environments
- BetterThanHumanRainbowAgent, 10M steps

#### `train_human_level.py` - Human-Level (268 dòng)
- AsyncVectorEnv + torch.compile
- BetterThanHumanRainbowAgent với attention + NoisyNets
- 5M steps, CosineAnnealingLR scheduler

### 2.3. Scripts Tiện Ích (`scripts/`)

| File | Chức năng | Dòng |
|------|-----------|------|
| `report_existing.py` | Đọc progress.json, in bảng tổng hợp | 59 |
| `instantiate_agents.py` | Kiểm tra 4 agent (Random, RuleBased, DQN, Rainbow) có thể act() | 71 |
| `render_gameplay.py` | Render gameplay ra file MP4 bằng OpenCV | 99 |
| `eval_checkpoints.py` | Đánh giá nhiều checkpoints | - |
| `benchmark_speed.py` | Benchmark training speed | - |

### 2.4. Kết Quả Phân Tích (`results/`)

| File | Chức năng |
|------|-----------|
| `plot_training.py` | Đọc metrics.json → vẽ eval_curves.png + loss_curves.png |
| `ablation_table.py` | Đọc progress.json → in bảng Markdown so sánh các runs |

### 2.5. Tests (`tests/`)

| File | Test |
|------|------|
| `test_agents_instantiate.py` | 4 tests: Random, RuleBased, DQN, Rainbow đều act() được với dummy state (4,84,84) |
| `test_config.py` | Test parse rainbow.yaml, kiểm tra keys và hyperparameters |
| `test_instantiate_exported.py` | Test agents từ package export |
| `test_rainbow_low_compute.py` | Test Rainbow trên cấu hình thấp |
| `test_better_than_human_agent.py` | Test BetterThanHuman agent |

---

## 3. Các Lệnh Sử Dụng

### 3.1. Cài Đặt

```bash
# Tạo virtualenv
python -m venv .venv && source .venv/bin/activate

# Cài dependencies (editable install)
pip install -e .

# Hoặc cài trực tiếp
pip install -r requirements.txt
```

### 3.2. Huấn Luyện

```bash
# Rainbow DQN cơ bản (YAML-driven, 1M steps, CPU)
python experiments/train.py -c configs/rainbow.yaml -o runs/rainbow_demo

# Rainbow gốc (1M steps, no reward clip)
python experiments/train_riverraid.py --steps 1000000

# Quick test (10K steps)
python experiments/train_riverraid.py --quick-test

# Big model (hidden_dim=512)
python experiments/train_riverraid.py --big-model --steps 2000000

# Resume từ checkpoint
python experiments/train_riverraid.py --resume checkpoints/rainbow/best.pt --steps 2000000

# CPU tối ưu (200K steps)
python experiments/train_cpu.py

# Rainbow nâng cao (200K steps, no clip)
python experiments/train_improved.py

# GPU + Vectorized (10M steps, 8 envs)
python experiments/train_optimized.py --steps 10000000 --envs 8

# Human-level (5M steps, 4 async envs)
python experiments/train_human_level.py --steps 5000000 --envs 4

# Human-level với attention
python experiments/train_optimized.py --attention --steps 10000000

# Quick test GPU
python experiments/train_optimized.py --quick-test
```

### 3.3. Đánh Giá

```bash
# Đánh giá checkpoint
python experiments/evaluate.py --checkpoint runs/rainbow_demo/best.pt --episodes 20

# In performance targets
python experiments/train_riverraid.py --targets
```

### 3.4. Scripts Tiện Ích (Không Cần GPU/Training)

```bash
# Xem kết quả checkpoints hiện có
python scripts/report_existing.py

# Kiểm tra tất cả agent có thể khởi tạo
python scripts/instantiate_agents.py

# Render gameplay video
python scripts/render_gameplay.py

# Đánh giá nhiều checkpoints
python scripts/eval_checkpoints.py

# Benchmark tốc độ
python scripts/benchmark_speed.py
```

### 3.5. Phân Tích & Trực Quan Hóa

```bash
# Vẽ đồ thị learning curves
python results/plot_training.py
python results/plot_training.py --paths logs/run1 logs/run2

# Bảng so sánh Ablation Study
python results/ablation_table.py
```

### 3.6. Tests

```bash
# Chạy tất cả tests
pytest tests/ -v

# Chạy test cụ thể
pytest tests/test_agents_instantiate.py -v
pytest tests/test_config.py -v
```

### 3.7. Ablation Study

```bash
# Huấn luyện từng biến thể
python experiments/train.py -c configs/rainbow.yaml -o runs/rainbow

# So sánh kết quả
python results/ablation_table.py
python results/plot_training.py --paths runs/dqn runs/rainbow
```

### 3.8. Notebook

```bash
jupyter notebook notebooks/quickstart.ipynb
```

### 3.9. Lint & Format

```bash
ruff check .
ruff format .
```

---

## 4. Kết Quả Mong Đợi

### 4.1. Benchmark Targets

| Metric | Giá trị | Ghi chú |
|--------|---------|---------|
| **Theoretical Max** | 1,000,000 | Giới hạn điểm tối đa của game |
| **Rainbow SOTA** (200M frames) | 20,675 | Kết quả tốt nhất từ paper gốc |
| **Human Expert** | 13,513 | Trung vị người chơi chuyên nghiệp |
| **DQN Nature** (2015) | 8,311 | Kết quả DQN ban đầu |
| **Rule-Based** | ~405 | Heuristic cơ bản |
| **Random Agent** | ~282 | Chọn hành động ngẫu nhiên |

### 4.2. Kết Quả Từng Loại Training

#### Rainbow DQN Cơ Bản (`train.py`, `train_riverraid.py`)
- **1M steps**: Thời gian ~30-60 phút (CPU), điểm ~400-600
- **Output**: `checkpoints/<run_name>/` chứa best.pt, final.pt, progress.json
- **Logs**: `logs/<run_name>/metrics.json` (loss, q_value, epsilon, eval/mean)

#### CPU Tối Ưu (`train_cpu.py`)
- **200K steps**: ~5-15 phút (CPU), 21 atoms (ít hơn → nhanh hơn)
- **Checkpoint**: `checkpoints/rainbow-cpu/best.pt`

#### Rainbow Nâng Cao (`train_improved.py`)
- **200K steps**: ~10-20 phút, 51 atoms, no reward clipping
- **Checkpoint**: `checkpoints/rainbow-improved/best.pt`

#### GPU + Vectorized (`train_optimized.py`)
- **10M steps**: ~2-4 giờ (GPU), 8 environments song song
- **Output**: best.pt, final.pt, metrics.json, summary.json
- **% Human Expert**: Phụ thuộc vào số steps và cấu hình

#### Human-Level (`train_human_level.py`)
- **5M steps**: ~1-3 giờ (GPU), AsyncVectorEnv + torch.compile
- **Output**: best.pt, final.pt, progress.json, summary.json
- **% Human Expert**: Mục tiêu vượt qua 13,513 điểm

### 4.3. File Output Khi Training

| File | Nội dung |
|------|----------|
| `checkpoints/<run>/best.pt` | Checkpoint tốt nhất (state_dict Q-network, target network, optimizer, steps, epsilon, beta) |
| `checkpoints/<run>/final.pt` | Checkpoint cuối cùng |
| `checkpoints/<run>/step_<N>.pt` | Checkpoint định kỳ |
| `checkpoints/<run>/progress.json` | Lịch sử eval: mean_reward, std, %human, %sota, timestamps |
| `logs/<run>/metrics.json` | Training metrics: loss, q_value, epsilon, eval/mean per step |
| `logs/<run>/train_output.log` | Log text đầy đủ (human-level) |
| `results/figures/eval_curves.png` | Đồ thị eval reward vs training steps |
| `results/figures/loss_curves.png` | Đồ thị training loss |
| `media/riverraid-best.mp4` | Video gameplay từ best checkpoint |

### 4.4. Định Dạng Output

#### Progress Report (in ra terminal khi eval)
```
===========================================================================
                        Progress Report
===========================================================================
  Metric                            Current       Best Ever         Target
---------------------------------------------------------------------------
  Raw Score                            549.0         588.0      1,000,000
  % of Theoretical Max (1M)         0.0549%       0.0588%     100.0000%
  % of Human Expert (13.5K)           4.06%         4.35%       100.00%
  % of Rainbow SOTA (20.7K)           2.66%         2.85%       100.00%
  % of DQN Nature (8.3K)              6.61%         7.08%       100.00%
===========================================================================
```

#### Summary Table (khi dùng `report_existing.py`)
```
====================================================================================================
Run                                Best    Steps     Mean  %Human   %SOTA  Time(s)
====================================================================================================
human-level-1720000000            588.0    17500    549.0    4.06    2.66     3600
rainbow-1m-1720000000             420.0   100000    380.0    2.81    1.84     1800
====================================================================================================
```

#### Ablation Table (Markdown)
```markdown
## Ablation Study - Results

| Run                             |    Steps |     Best | Final Mean | %Human | %SOTA | Time (s) |
|---------------------------------|----------|----------|------------|--------|-------|----------|
| rainbow                         |  1000000 |    588.0 |      549.0 |   4.06 |  2.66 |     3600 |
```

### 4.5. Workflow Tóm Tắt

```
[1] Cài đặt: pip install -e .
         ↓
[2] Huấn luyện: python experiments/train.py -c configs/rainbow.yaml -o runs/exp1
         ↓
[3] Kiểm tra kết quả: python scripts/report_existing.py
         ↓
[4] Vẽ đồ thị: python results/plot_training.py --paths runs/exp1
         ↓
[5] Đánh giá chi tiết: python experiments/evaluate.py --checkpoint runs/exp1/best.pt --episodes 50
         ↓
[6] Render video: python scripts/render_gameplay.py
```

---

## 5. Kiến Trúc Mô Hình

### 5.1. Rainbow DQN Architecture

```
Input (4, 84, 84) [4 frames stacked]
    ↓
Conv2d(4→32, 8x8, stride=4) + ReLU
    ↓
Conv2d(32→64, 4x4, stride=2) + ReLU
    ↓
Conv2d(64→64, 3x3, stride=1) + ReLU
    ↓
Flatten → 3136 (64*7*7)
    ↓
┌─────────────────┬─────────────────┐
│  Value Stream   │ Advantage Stream│
│  Linear→256     │  Linear→256     │
│  Linear→51 atoms│  Linear→6*51    │
└─────────────────┴─────────────────┘
    ↓
V(s) + A(s,a) - mean(A)
    ↓
Softmax over 51 atoms → Distribution Z(s,a)
    ↓
Q(s,a) = sum(z * p(z))
```

### 5.2. Replay Buffer Flow

```
Environment Step → push(state, action, reward, next_state, done)
    ↓
NStepReplayBuffer (accumulates n-step returns)
    ↓
PrioritizedReplayBuffer (sum-tree sampling proportional to |TD-error|^alpha)
    ↓
sample(batch_size, beta) → (states, actions, rewards, next_states, dones, indices, weights)
    ↓
Agent.update() → compute loss → backward → update_priorities()
```

---

## 6. Phụ Thuộc Chính

| Package | Phiên bản | Mục đích |
|---------|-----------|----------|
| torch | >=2.0 | Deep learning framework |
| gymnasium[atari] | * | Atari environment |
| ale-py | >=0.8 | Arcade Learning Environment |
| numpy | * | Numerical computing |
| pyyaml | * | Đọc config YAML |
| matplotlib | * | Vẽ đồ thị (dev) |
| pytest | * | Testing (dev) |
| ruff | * | Linting (dev) |
| jupyter | * | Notebook (dev) |
| opencv-python | * | Render video gameplay |
