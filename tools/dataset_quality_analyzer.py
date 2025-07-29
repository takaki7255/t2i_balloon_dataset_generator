"""
データセット品質評価と継続的改善のための分析ツール
"""

import json
import re
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from pathlib import Path
import os

class DatasetQualityAnalyzer:
    def __init__(self, dataset_path):
        self.dataset_path = Path(dataset_path)
        self.config_path = self.dataset_path / "config.json"
        self.train_log_path = self.dataset_path / "train_composition_log.txt"
        self.val_log_path = self.dataset_path / "val_composition_log.txt"
        
    def load_config(self):
        """設定ファイルを読み込み"""
        if self.config_path.exists():
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None
    
    def parse_log_file(self, log_file_path):
        """ログファイルを解析して統計情報を抽出"""
        balloon_counts = []
        scale_values = []
        scale_ratios = []
        crop_efficiencies = []
        balloon_sizes = []
        
        if not log_file_path.exists():
            return {}, [], [], [], [], []
            
        with open(log_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 各画像の情報を抽出
        images = content.split('画像 ')[1:]
        
        for image_text in images:
            lines = image_text.split('\n')
            
            # 配置した吹き出し数を抽出
            for line in lines:
                if '配置した吹き出し数:' in line:
                    count = int(line.split(':')[1].strip())
                    balloon_counts.append(count)
                    break
            
            # 各吹き出しの詳細情報を抽出
            for line in lines:
                if 'スケール値:' in line:
                    scale_val = float(line.split(':')[1].strip())
                    scale_values.append(scale_val)
                elif '画面幅比:' in line:
                    ratio_val = float(line.split(':')[1].strip())
                    scale_ratios.append(ratio_val)
                elif 'クロップ効率:' in line:
                    eff_val = float(line.split(':')[1].strip())
                    crop_efficiencies.append(eff_val)
                elif '最終サイズ:' in line:
                    size_str = line.split(':')[1].strip()
                    if 'x' in size_str:
                        w, h = map(int, size_str.split('x'))
                        balloon_sizes.append((w, h))
        
        stats = {
            'balloon_count_mean': np.mean(balloon_counts) if balloon_counts else 0,
            'balloon_count_range': (min(balloon_counts), max(balloon_counts)) if balloon_counts else (0, 0),
            'balloon_count_distribution': dict(Counter(balloon_counts)),
            'scale_mean': np.mean(scale_values) if scale_values else 0,
            'scale_range': (min(scale_values), max(scale_values)) if scale_values else (0, 0),
            'ratio_mean': np.mean(scale_ratios) if scale_ratios else 0,
            'ratio_range': (min(scale_ratios), max(scale_ratios)) if scale_ratios else (0, 0),
            'crop_efficiency_mean': np.mean(crop_efficiencies) if crop_efficiencies else 0,
            'total_balloons': len(scale_values),
            'total_images': len(balloon_counts)
        }
        
        return stats, balloon_counts, scale_values, scale_ratios, crop_efficiencies, balloon_sizes
    
    def generate_report(self):
        """包括的な品質レポートを生成"""
        config = self.load_config()
        if not config:
            print("⚠️ 設定ファイルが見つかりません")
            return
            
        print("=" * 80)
        print(f"📊 データセット品質分析レポート")
        print(f"データセット: {self.dataset_path.name}")
        print(f"生成時刻: {config.get('timestamp', 'N/A')}")
        print("=" * 80)
        
        # 設定情報の表示
        cfg = config.get('config', {})
        print(f"\n🔧 設定情報:")
        print(f"  吹き出し個数範囲: {cfg.get('NUM_BALLOONS_RANGE', 'N/A')}")
        print(f"  スケール設定: mode={cfg.get('SCALE_MODE', 'N/A')}, mean={cfg.get('SCALE_MEAN', 'N/A')}")
        print(f"  スケール範囲: {cfg.get('SCALE_CLIP', 'N/A')}")
        print(f"  目標画像数: {cfg.get('TARGET_TOTAL_IMAGES', 'N/A')}")
        
        # Train データの分析
        train_stats, train_counts, train_scales, train_ratios, train_effs, train_sizes = self.parse_log_file(self.train_log_path)
        
        if train_stats:
            print(f"\n📈 TRAIN データ分析:")
            print(f"  生成画像数: {train_stats['total_images']}")
            print(f"  総吹き出し数: {train_stats['total_balloons']}")
            print(f"  吹き出し個数: 平均{train_stats['balloon_count_mean']:.2f}個 (範囲: {train_stats['balloon_count_range'][0]}-{train_stats['balloon_count_range'][1]})")
            print(f"  画面幅比: 平均{train_stats['ratio_mean']:.3f} ({train_stats['ratio_mean']*100:.1f}%)")
            print(f"  クロップ効率: 平均{train_stats['crop_efficiency_mean']:.3f}")
        
        # Val データの分析
        val_stats, val_counts, val_scales, val_ratios, val_effs, val_sizes = self.parse_log_file(self.val_log_path)
        
        if val_stats:
            print(f"\n📊 VAL データ分析:")
            print(f"  生成画像数: {val_stats['total_images']}")
            print(f"  総吹き出し数: {val_stats['total_balloons']}")
            print(f"  吹き出し個数: 平均{val_stats['balloon_count_mean']:.2f}個 (範囲: {val_stats['balloon_count_range'][0]}-{val_stats['balloon_count_range'][1]})")
            print(f"  画面幅比: 平均{val_stats['ratio_mean']:.3f} ({val_stats['ratio_mean']*100:.1f}%)")
            print(f"  クロップ効率: 平均{val_stats['crop_efficiency_mean']:.3f}")
        
        # 全体統計
        all_counts = train_counts + val_counts
        all_ratios = train_ratios + val_ratios
        
        if all_counts and all_ratios:
            print(f"\n🌟 全体統計:")
            print(f"  全体画像数: {len(all_counts)}")
            print(f"  全体吹き出し数: {len(all_ratios)}")
            print(f"  吹き出し個数: 平均{np.mean(all_counts):.2f}個")
            print(f"  画面幅比: 平均{np.mean(all_ratios):.3f} ({np.mean(all_ratios)*100:.1f}%)")
            
            # 実際の統計との比較
            print(f"\n🔍 実際のマンガ統計との比較:")
            print(f"  実際の統計: 平均12.26個、中央値12個")
            print(f"  生成データ: 平均{np.mean(all_counts):.2f}個")
            
            diff = abs(np.mean(all_counts) - 12.26)
            if diff < 0.5:
                print(f"  → ✅ 優秀 (差: {diff:.2f}個)")
            elif diff < 1.0:
                print(f"  → ⚠️ 良好 (差: {diff:.2f}個)")
            else:
                print(f"  → ❌ 要改善 (差: {diff:.2f}個)")
        
        # データセット構造の確認
        print(f"\n📁 データセット構造:")
        train_dir = self.dataset_path / "train"
        val_dir = self.dataset_path / "val"
        
        if train_dir.exists():
            train_imgs = len(list(train_dir.glob("*.png")))
            train_masks = len(list((train_dir / "masks").glob("*.png"))) if (train_dir / "masks").exists() else 0
            print(f"  train: {train_imgs}画像, {train_masks}マスク")
            
        if val_dir.exists():
            val_imgs = len(list(val_dir.glob("*.png")))
            val_masks = len(list((val_dir / "masks").glob("*.png"))) if (val_dir / "masks").exists() else 0
            print(f"  val: {val_imgs}画像, {val_masks}マスク")
    
    def plot_distributions(self):
        """分布の可視化"""
        train_stats, train_counts, train_scales, train_ratios, _, _ = self.parse_log_file(self.train_log_path)
        val_stats, val_counts, val_scales, val_ratios, _, _ = self.parse_log_file(self.val_log_path)
        
        if not train_counts and not val_counts:
            print("⚠️ プロット用のデータが不足しています")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Dataset Quality Analysis - {self.dataset_path.name}', fontsize=16)
        
        # 吹き出し個数分布
        all_counts = train_counts + val_counts
        if all_counts:
            axes[0, 0].hist(all_counts, bins=range(min(all_counts), max(all_counts)+2), alpha=0.7, edgecolor='black')
            axes[0, 0].axvline(12.26, color='red', linestyle='--', label='Real manga avg (12.26)')
            axes[0, 0].axvline(np.mean(all_counts), color='blue', linestyle='-', label=f'Generated avg ({np.mean(all_counts):.2f})')
            axes[0, 0].set_title('Balloon Count Distribution')
            axes[0, 0].set_xlabel('Number of Balloons')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].legend()
        
        # 画面幅比分布
        all_ratios = train_ratios + val_ratios
        if all_ratios:
            axes[0, 1].hist(all_ratios, bins=30, alpha=0.7, edgecolor='black')
            axes[0, 1].axvline(np.mean(all_ratios), color='blue', linestyle='-', label=f'Avg ({np.mean(all_ratios):.3f})')
            axes[0, 1].set_title('Screen Width Ratio Distribution')
            axes[0, 1].set_xlabel('Width Ratio')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].legend()
        
        # スケール値分布
        all_scales = train_scales + val_scales
        if all_scales:
            axes[1, 0].hist(all_scales, bins=30, alpha=0.7, edgecolor='black')
            axes[1, 0].axvline(np.mean(all_scales), color='blue', linestyle='-', label=f'Avg ({np.mean(all_scales):.3f})')
            axes[1, 0].set_title('Scale Value Distribution')
            axes[1, 0].set_xlabel('Scale Value')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].legend()
        
        # Train vs Val 比較
        if train_counts and val_counts:
            axes[1, 1].hist([train_counts, val_counts], bins=range(min(all_counts), max(all_counts)+2), 
                           alpha=0.7, label=['Train', 'Val'], edgecolor='black')
            axes[1, 1].set_title('Train vs Val Balloon Counts')
            axes[1, 1].set_xlabel('Number of Balloons')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].legend()
        
        plt.tight_layout()
        output_path = self.dataset_path / "quality_analysis.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"📊 分布グラフを保存: {output_path}")
        plt.show()

def main():
    """メイン処理"""
    dataset_path = "../syn_mihiraki300_dataset"
    
    if not os.path.exists(dataset_path):
        print(f"❌ データセットが見つかりません: {dataset_path}")
        return
    
    analyzer = DatasetQualityAnalyzer(dataset_path)
    analyzer.generate_report()
    analyzer.plot_distributions()

if __name__ == "__main__":
    main()
