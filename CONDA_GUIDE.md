# Anaconda環境での実験実行ガイド

## 🚀 クイックスタート

### 1. Conda環境を確認

```powershell
# Anaconda Promptを開く
conda env list
```

出力例:
```
# conda environments:
#
base                  *  C:\Users\user\anaconda3
myenv                    C:\Users\user\anaconda3\envs\myenv
pytorch_env              C:\Users\user\anaconda3\envs\pytorch_env
```

### 2. スクリプトを実行

#### 方法A: バッチファイル（最も簡単）

```batch
REM Anaconda Promptで実行
cd C:\path\to\t2i_balloon_dataset_generator

REM 全データセットで実行
.\scripts\run_experiments_conda.bat myenv

REM 特定データセットのみ
.\scripts\run_experiments_conda.bat myenv syn500-corner
```

#### 方法B: PowerShellスクリプト（柔軟）

```powershell
# PowerShellまたはAnaconda Promptで実行
.\scripts\run_experiments_conda.ps1 -CondaEnv myenv

# オプション付き
.\scripts\run_experiments_conda.ps1 -CondaEnv myenv -Datasets syn500-corner -TrainOnly
```

#### 方法C: クイック実行（単一データセット）

```powershell
# 最速で1つのデータセットをテスト
.\scripts\quick_train_test_conda.ps1 -CondaEnv myenv -Dataset syn500-corner
```

---

## 📋 よくある使用例

### 例1: 初めての実験（1つのデータセット）

```powershell
# まず小さいデータセットで試す
.\scripts\quick_train_test_conda.ps1 -CondaEnv myenv -Dataset syn200-corner
```

### 例2: すべてのデータセットで完全実験

```batch
REM バッチファイルで実行（簡単）
.\scripts\run_experiments_conda.bat myenv

REM またはPowerShell
.\scripts\run_experiments_conda.ps1 -CondaEnv myenv
```

### 例3: 学習だけ実行（テストは後で）

```powershell
.\scripts\run_experiments_conda.ps1 -CondaEnv myenv -TrainOnly
```

### 例4: 学習済みモデルでテストのみ

```powershell
.\scripts\run_experiments_conda.ps1 -CondaEnv myenv -TestOnly
```

### 例5: Wandbを使わずに実験

```powershell
# ローカルでのみ結果を保存
.\scripts\run_experiments_conda.ps1 -CondaEnv myenv -NoWandb
```

### 例6: 特定の2つのデータセットだけ

```powershell
.\scripts\run_experiments_conda.ps1 -CondaEnv myenv -Datasets "syn500-corner,syn1000-corner"
```

---

## 🔧 トラブルシューティング

### Q: "conda: コマンドが見つかりません"

**A:** Anaconda Promptを使用してください。

1. スタートメニューから「Anaconda Prompt」を検索
2. Anaconda Promptを開く
3. プロジェクトディレクトリに移動
   ```batch
   cd C:\Users\ttaka\manga\t2i_balloon_dataset_generator
   ```
4. スクリプトを実行

### Q: "Conda環境が見つかりません"

**A:** 環境名を確認してください。

```powershell
# 利用可能な環境を表示
conda env list

# 正しい環境名を使用
.\scripts\run_experiments_conda.ps1 -CondaEnv <正しい環境名>
```

### Q: PowerShell実行ポリシーエラー

**A:** 実行ポリシーを変更してください。

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Q: メモリ不足エラー

**A:** バッチサイズを減らしてください。

スクリプト内の設定を編集：
```powershell
batch = 4  # 8 → 4 に変更
```

またはクイック実行で指定：
```powershell
.\scripts\quick_train_test_conda.ps1 -CondaEnv myenv -Dataset syn500-corner -Batch 4
```

---

## 📁 実行後の確認

### モデルファイル

```
balloon_models/
├── syn200-corner-unet-01.pt
├── syn500-corner-unet-01.pt
├── syn750-corner-unet-01.pt
└── syn1000-corner-unet-01.pt
```

### 実験結果

```
experiment_results/
├── experiment_log_20251028_143000.txt  ← 実験ログ
├── syn200-corner-unet-01/
│   ├── evaluation_results.json
│   ├── evaluation_summary.txt
│   └── comparisons/
└── ...
```

### ログの確認

```powershell
# 最新のログを表示
Get-Content experiment_results\experiment_log_*.txt -Tail 50
```

---

## 💡 ヒント

### 夜間実行

タスクスケジューラに登録：
```powershell
# タスク作成
$action = New-ScheduledTaskAction -Execute "cmd.exe" -Argument "/c cd C:\path\to\project && scripts\run_experiments_conda.bat myenv"
$trigger = New-ScheduledTaskTrigger -Daily -At "2:00AM"
Register-ScheduledTask -TaskName "UNet実験" -Action $action -Trigger $trigger
```

### 進捗確認

別のPowerShellウィンドウで：
```powershell
# GPU使用率を監視
nvidia-smi -l 5

# ログをリアルタイムで表示
Get-Content experiment_results\experiment_log_*.txt -Wait
```

### 緊急停止

実験を中断したい場合：
- `Ctrl + C` を押す
- 現在のエポックが完了してから停止します
- 途中までの結果は保存されます

---

## 📞 サポート

問題が解決しない場合：

1. `experiment_log_*.txt` の内容を確認
2. Conda環境のパッケージを確認: `conda list`
3. GPUが認識されているか確認: `nvidia-smi`
4. PyTorchでGPUが使えるか確認:
   ```powershell
   conda run -n myenv python -c "import torch; print(torch.cuda.is_available())"
   ```
