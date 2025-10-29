# 実験自動化スクリプト

このフォルダには、複数のデータセットでU-Net学習とテストを自動実行するスクリプトが含まれています。

## 📁 ファイル一覧

### 標準Python環境用

| ファイル | 説明 | 環境 |
|---------|------|------|
| `run_experiments.ps1` | 複数データセット自動実験（Windows） | Python |
| `run_experiments.sh` | 複数データセット自動実験（Linux/Mac） | Python |
| `quick_train_test.ps1` | 単一データセット高速実験（Windows） | Python |

### Anaconda/Conda環境用

| ファイル | 説明 | 環境 |
|---------|------|------|
| `run_experiments_conda.ps1` | 複数データセット自動実験（PowerShell） | Conda |
| `run_experiments_conda.bat` | 複数データセット自動実験（バッチ） | Conda |
| `quick_train_test_conda.ps1` | 単一データセット高速実験 | Conda |

---

## 🚀 クイックスタート

### Windows + Anaconda（推奨）

```powershell
# プロジェクトルートで実行
cd C:\Users\ttaka\manga\t2i_balloon_dataset_generator

# Anaconda Promptまたは通常のPowerShell
.\scripts\run_experiments_conda.ps1 -CondaEnv myenv

# バッチファイル版（Anaconda Prompt）
.\scripts\run_experiments_conda.bat myenv
```

### Windows + 標準Python

```powershell
.\scripts\run_experiments.ps1
```

### Linux/Mac

```bash
chmod +x ./scripts/run_experiments.sh
./scripts/run_experiments.sh
```

---

## 📖 使用方法

### 🐍 Anaconda環境（推奨）

#### 全データセットで実行
```powershell
.\scripts\run_experiments_conda.ps1 -CondaEnv myenv
```

#### 特定データセットのみ
```powershell
.\scripts\run_experiments_conda.ps1 -CondaEnv myenv -Datasets syn500-corner
```

#### 学習のみ
```powershell
.\scripts\run_experiments_conda.ps1 -CondaEnv myenv -TrainOnly
```

#### テストのみ
```powershell
.\scripts\run_experiments_conda.ps1 -CondaEnv myenv -TestOnly
```

#### クイック実行（単一データセット）
```powershell
.\scripts\quick_train_test_conda.ps1 -CondaEnv myenv -Dataset syn500-corner
```

### 🪟 標準Python環境

#### 全データセットで実行
```powershell
.\scripts\run_experiments.ps1
```

#### 特定データセットのみ
```powershell
.\scripts\run_experiments.ps1 -Datasets "syn200-corner,syn500-corner"
```

#### クイック実行
```powershell
.\scripts\quick_train_test.ps1 -Dataset syn500-corner
```

### 🐧 Linux/Mac

```bash
# 全データセット
./scripts/run_experiments.sh

# 特定データセット
./scripts/run_experiments.sh --datasets=syn500-corner

# オプション
./scripts/run_experiments.sh --train-only --no-wandb
```

---

## 🎯 対象データセット

デフォルトで以下のデータセットが登録されています：

| データセット | パス | エポック | バッチ | Patience |
|------------|------|---------|--------|----------|
| syn200-corner | ./balloon_dataset/syn200_dataset | 100 | 8 | 15 |
| syn500-corner | ./balloon_dataset/syn500_dataset | 100 | 8 | 15 |
| syn750-corner | ./balloon_dataset/syn750_dataset | 100 | 8 | 15 |
| syn1000-corner | ./balloon_dataset/syn1000_dataset | 100 | 8 | 15 |

---

## 📦 出力

### モデルファイル
```
../balloon_models/
├── syn200-corner-unet-01.pt
├── syn500-corner-unet-01.pt
└── ...
```

### 実験結果
```
../experiment_results/
├── experiment_log_YYYYMMDD_HHMMSS.txt
├── syn200-corner-unet-01/
│   ├── evaluation_results.json
│   ├── evaluation_summary.txt
│   ├── images/
│   ├── predicts/
│   └── comparisons/
└── ...
```

---

## 🔧 カスタマイズ

スクリプト内の設定を編集してパラメータを変更できます：

```powershell
# PowerShell (.ps1)
$DATASET_CONFIGS = @(
    @{
        name = "my-dataset"
        root = "./my_data"
        epochs = 150
        batch = 16
        patience = 20
    }
)
```

---

## 📚 詳細ドキュメント

- **`../EXPERIMENT_AUTOMATION.md`** - 詳細な使用方法とオプション
- **`../CONDA_GUIDE.md`** - Anaconda環境での使用ガイド

---

## 💡 ヒント

### Anaconda Promptで実行する場合

1. スタートメニューから「Anaconda Prompt」を起動
2. プロジェクトディレクトリに移動
   ```batch
   cd C:\Users\ttaka\manga\t2i_balloon_dataset_generator
   ```
3. バッチファイルを実行（最も簡単）
   ```batch
   .\scripts\run_experiments_conda.bat myenv
   ```

### PowerShell実行ポリシーエラー

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### メモリ不足の場合

バッチサイズを減らす：
```powershell
.\scripts\quick_train_test_conda.ps1 -CondaEnv myenv -Dataset syn500-corner -Batch 4
```

---

## 🆘 トラブルシューティング

### "conda: コマンドが見つかりません"
→ Anaconda Promptを使用してください

### "Conda環境が見つかりません"
→ 環境名を確認: `conda env list`

### スクリプトが実行できない
→ 実行権限を付与:
```bash
# Linux/Mac
chmod +x ./scripts/*.sh
```

---

## 📝 ライセンス

MIT License
