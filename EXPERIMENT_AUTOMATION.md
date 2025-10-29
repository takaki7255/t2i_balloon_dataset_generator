# 実験自動化スクリプト

複数のデータセットでU-Net学習とテストを自動的に実行するスクリプト群です。

## スクリプト一覧

すべてのスクリプトは `scripts/` フォルダに配置されています。

### 1. `scripts/run_experiments.ps1` / `scripts/run_experiments.sh`
複数データセットで学習とテストを一括実行（標準Python環境）

### 2. `scripts/run_experiments_conda.ps1` / `scripts/run_experiments_conda.bat`
**Anaconda環境**で複数データセットを一括実行

### 3. `scripts/quick_train_test.ps1`
単一データセットで素早く実験（Windows専用、標準Python）

### 4. `scripts/quick_train_test_conda.ps1`
単一データセットで素早く実験（Windows専用、**Anaconda環境**）

---

## 使用方法

### 🐍 Anaconda環境で実行（推奨）

#### Anaconda Prompt / PowerShell

```powershell
# 全データセットで実行
.\scripts\run_experiments_conda.ps1 -CondaEnv myenv

# 特定のデータセットのみ
.\scripts\run_experiments_conda.ps1 -CondaEnv myenv -Datasets syn500-corner

# 複数データセット
.\scripts\run_experiments_conda.ps1 -CondaEnv myenv -Datasets "syn200-corner,syn500-corner"

# 学習のみ
.\scripts\run_experiments_conda.ps1 -CondaEnv myenv -TrainOnly

# テストのみ
.\scripts\run_experiments_conda.ps1 -CondaEnv myenv -TestOnly

# Wandb無効化
.\scripts\run_experiments_conda.ps1 -CondaEnv myenv -NoWandb
```

#### バッチファイル（Anaconda Prompt）

```batch
REM 全データセットで実行
.\scripts\run_experiments_conda.bat myenv

REM 特定データセット
.\scripts\run_experiments_conda.bat myenv syn500-corner

REM 複数データセット
.\scripts\run_experiments_conda.bat myenv syn200-corner,syn500-corner
```

#### クイック実行（単一データセット）

```powershell
# 学習+テスト
.\scripts\quick_train_test_conda.ps1 -CondaEnv myenv -Dataset syn500-corner

# 学習のみ
.\scripts\quick_train_test_conda.ps1 -CondaEnv myenv -Dataset syn500-corner -SkipTest

# テストのみ
.\scripts\quick_train_test_conda.ps1 -CondaEnv myenv -Dataset syn500-corner -SkipTrain

# カスタムパラメータ
.\scripts\quick_train_test_conda.ps1 -CondaEnv myenv -Dataset syn500-corner -Epochs 150 -Batch 16
```

---

### Windows (PowerShell) - 標準Python環境

#### 全データセットで実行
```powershell
.\scripts\run_experiments.ps1
```

#### 学習のみ実行
```powershell
.\scripts\run_experiments.ps1 -TrainOnly
```

#### テストのみ実行
```powershell
.\scripts\run_experiments.ps1 -TestOnly
```

#### Wandb無効化
```powershell
.\scripts\run_experiments.ps1 -NoWandb
```

#### 特定のデータセットのみ実行
```powershell
# 単一データセット
.\scripts\run_experiments.ps1 -Datasets syn500-corner

# 複数データセット（カンマ区切り）
.\scripts\run_experiments.ps1 -Datasets "syn200-corner,syn500-corner,syn1000-corner"
```

#### オプション組み合わせ
```powershell
.\scripts\run_experiments.ps1 -TrainOnly -NoWandb -Datasets "syn500-corner,syn750-corner"
```

### クイック実行（単一データセット）

```powershell
# 学習+テスト
.\scripts\quick_train_test.ps1 -Dataset syn500-corner

# 学習のみ
.\scripts\quick_train_test.ps1 -Dataset syn500-corner -SkipTest

# テストのみ
.\scripts\quick_train_test.ps1 -Dataset syn500-corner -SkipTrain

# カスタムパラメータ
.\scripts\quick_train_test.ps1 -Dataset syn500-corner -Epochs 150 -Batch 16 -Patience 20
```

---

### Linux / Mac (Bash)

#### スクリプトに実行権限を付与
```bash
chmod +x scripts/run_experiments.sh
```

#### 全データセットで実行
```bash
./scripts/run_experiments.sh
```

#### 学習のみ実行
```bash
./scripts/run_experiments.sh --train-only
```

#### テストのみ実行
```bash
./scripts/run_experiments.sh --test-only
```

#### Wandb無効化
```bash
./scripts/run_experiments.sh --no-wandb
```

#### 特定のデータセットのみ実行
```bash
# 単一データセット
./scripts/run_experiments.sh --datasets=syn500-corner

# 複数データセット（カンマ区切り）
./scripts/run_experiments.sh --datasets=syn200-corner,syn500-corner,syn1000-corner
```

#### オプション組み合わせ
```bash
./scripts/run_experiments.sh --train-only --no-wandb --datasets=syn500-corner,syn750-corner
```

#### ヘルプ表示
```bash
./scripts/run_experiments.sh --help
```

---

## Conda環境のセットアップ

### 新しいConda環境を作成

```bash
# Python 3.11環境を作成
conda create -n myenv python=3.11 -y

# 環境をアクティベート
conda activate myenv

# 必要なパッケージをインストール
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy pillow scikit-learn matplotlib tqdm wandb
```

### 既存環境の確認

```bash
# 利用可能なConda環境を表示
conda env list

# 現在の環境を確認
conda info --envs
```

---

## 設定

### デフォルトパラメータ

各スクリプト内で以下のパラメータが設定されています：

| データセット | ルートパス | エポック | バッチサイズ | Patience |
|------------|-----------|---------|------------|----------|
| syn200-corner | ./balloon_dataset/syn200_dataset | 100 | 8 | 15 |
| syn500-corner | ./balloon_dataset/syn500_dataset | 100 | 8 | 15 |
| syn750-corner | ./balloon_dataset/syn750_dataset | 100 | 8 | 15 |
| syn1000-corner | ./balloon_dataset/syn1000_dataset | 100 | 8 | 15 |

### カスタマイズ

スクリプト内の以下の変数を編集して設定を変更できます：

```powershell
# PowerShell (.ps1)
$DATASET_CONFIGS = @(
    @{
        name = "syn200-corner"
        root = "./balloon_dataset/syn200_dataset"
        epochs = 100
        batch = 8
        patience = 15
    },
    # ... 追加設定
)
```

```bash
# Bash (.sh)
declare -a DATASET_NAMES=("syn200-corner" "syn500-corner" ...)
declare -a DATASET_ROOTS=("./balloon_dataset/syn200_dataset" ...)
declare -a DATASET_EPOCHS=(100 100 ...)
declare -a DATASET_BATCH=(8 8 ...)
declare -a DATASET_PATIENCE=(15 15 ...)
```

---

## 出力

### ディレクトリ構造

```
balloon_models/          # 学習済みモデル
  ├── syn200-corner-unet-01.pt
  ├── syn500-corner-unet-01.pt
  └── ...

experiment_results/      # テスト結果
  ├── experiment_log_YYYYMMDD_HHMMSS.txt  # 実験ログ
  ├── syn200-corner-unet-01/
  │   ├── evaluation_results.json
  │   ├── evaluation_summary.txt
  │   ├── images/
  │   ├── predicts/
  │   └── comparisons/
  └── ...
```

### ログファイル

実験の詳細ログは `experiment_results/experiment_log_YYYYMMDD_HHMMSS.txt` に保存されます。

各ステップのタイムスタンプ、成功/失敗、エラー情報が記録されます。

---

## トラブルシューティング

### PowerShell実行ポリシーエラー

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Pythonが見つからない

環境変数のPATHにPythonが含まれているか確認：
```powershell
# PowerShell
python --version

# Bash
python3 --version
```

### メモリ不足

バッチサイズを減らす：
```powershell
# スクリプト内のbatch設定を変更
batch = 4  # 8 → 4
```

---

## 応用例

### 夜間バッチ実行（Windows）

タスクスケジューラで設定：
```powershell
# 毎日午前2時に実行
$action = New-ScheduledTaskAction -Execute "PowerShell.exe" -Argument "-File C:\path\to\scripts\run_experiments.ps1 -NoWandb"
$trigger = New-ScheduledTaskTrigger -Daily -At 2am
Register-ScheduledTask -Action $action -Trigger $trigger -TaskName "UNetExperiments"
```

### 夜間バッチ実行（Linux）

crontabで設定：
```bash
# 毎日午前2時に実行
0 2 * * * cd /path/to/project && ./scripts/run_experiments.sh --no-wandb >> /path/to/logs/cron.log 2>&1
```

### 異なるハイパーパラメータで複数実行

スクリプトをコピーして設定を変更：
```powershell
# run_experiments_lr5e5.ps1 (学習率を変更)
# スクリプト内で --lr 5e-5 を追加
```

---

## 注意事項

- **実行時間**: 各データセット100エポックで約1-3時間（GPU性能に依存）
- **ディスク容量**: モデルファイルと結果で約500MB-1GB必要
- **メモリ**: 最低8GB RAM、6GB VRAM推奨
- **中断**: Ctrl+C で安全に停止可能（現在のエポックは完了）

---

## ライセンス

MIT License
