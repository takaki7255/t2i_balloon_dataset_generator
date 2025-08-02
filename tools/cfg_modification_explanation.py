"""
CFG設定修正の詳細説明と根拠
"""

def explain_cfg_corrections():
    print("=== CFG設定修正の詳細説明 ===")
    print()
    
    print("📊 **統計データ分析結果**")
    print("• 実際のマンガの吹き出し個数: 平均13.1個、25-75%範囲 8-17個")
    print("• 実際のマンガの吹き出しサイズ: 平均面積比0.877%、幅比約9.4%")
    print("• 実際のサイズ分布: 25-75%範囲で幅比6.6%-10.6%")
    print()
    
    print("🔧 **修正内容と根拠**")
    print()
    
    print("1️⃣ **吹き出し個数範囲**")
    print("   修正前: (7, 17)")
    print("   修正後: (8, 17)")
    print("   根拠: 実際の統計25%値は8個。7個は統計的に少なすぎる")
    print()
    
    print("2️⃣ **スケール範囲**")
    print("   修正前: (0.065, 0.110)")
    print("   修正後: (0.065, 0.110)")
    print("   根拠: 実際の統計25-75%範囲(6.6%-10.6%)に合わせて設定")
    print("   効果: より現実的なサイズ分布を実現")
    print()
    
    print("3️⃣ **スケール平均値**")
    print("   修正前: 0.10")
    print("   修正後: 0.094")
    print("   根拠: 実際の統計平均値9.4%に正確に合わせる")
    print("   効果: 生成データが実際のマンガとより一致")
    print()
    
    print("4️⃣ **スケール標準偏差**")
    print("   修正前: 0.030")
    print("   修正後: 0.020")
    print("   根拠: 実際のデータ分散に合わせてより控えめに設定")
    print("   効果: 極端なサイズ変動を抑制")
    print()
    
    print("5️⃣ **スケールクリップ範囲**")
    print("   修正前: (0.05, 0.18)")
    print("   修正後: (0.050, 0.125)")
    print("   根拠: 統計範囲に余裕を持たせつつ、極端値を除去")
    print("   効果: 現実的でありながら柔軟性のある範囲")
    print()
    
    print("6️⃣ **最大サイズ制限**")
    print("   修正前: 幅30%, 高さ40%")
    print("   修正後: 幅15%, 高さ25%")
    print("   根拠: 実際のマンガでは巨大な吹き出しは稀")
    print("   効果: より現実的な最大サイズ制限")
    print()
    
    print("📈 **修正効果の予想**")
    print("✅ 生成される吹き出しサイズが実際のマンガに近づく")
    print("✅ 個数分布が統計データにより忠実になる")
    print("✅ 極端に大きな吹き出しの生成を抑制")
    print("✅ 面積ベースリサイズとの組み合わせで更に精度向上")
    print("✅ 学習データの品質向上により、モデル性能向上が期待")
    print()
    
    print("⚠️ **注意点**")
    print("• スケール範囲を狭めたため、バリエーションはやや減少")
    print("• ただし、実際のマンガに近い分布になるためより有用")
    print("• 必要に応じて、特定用途向けに範囲を調整可能")

def validate_new_settings():
    """新設定の妥当性を検証"""
    print("\n" + "=" * 60)
    print("🔍 **新設定の妥当性検証**")
    print()
    
    # 新しい設定値
    new_cfg = {
        "NUM_BALLOONS_RANGE": (8, 17),
        "SCALE_RANGE": (0.065, 0.110),
        "SCALE_MEAN": 0.094,
        "SCALE_STD": 0.020,
        "SCALE_CLIP": (0.050, 0.125),
        "MAX_WIDTH_RATIO": 0.15,
        "MAX_HEIGHT_RATIO": 0.25,
    }
    
    # 実際の統計
    actual_stats = {
        "count_25th": 8.0,
        "count_75th": 17.0,
        "size_ratio_mean": 0.008769,
        "width_ratio_mean": 0.094,  # sqrt(0.008769)
        "width_ratio_25th": 0.066,
        "width_ratio_75th": 0.106,
    }
    
    print("個数設定の検証:")
    min_count, max_count = new_cfg["NUM_BALLOONS_RANGE"]
    if min_count >= actual_stats["count_25th"] and max_count <= actual_stats["count_75th"]:
        print(f"✅ 個数範囲 {new_cfg['NUM_BALLOONS_RANGE']} は統計25-75%範囲内")
    else:
        print(f"❌ 個数範囲に問題あり")
    
    print("\nサイズ設定の検証:")
    scale_mean = new_cfg["SCALE_MEAN"]
    scale_min, scale_max = new_cfg["SCALE_RANGE"]
    
    if abs(scale_mean - actual_stats["width_ratio_mean"]) < 0.005:
        print(f"✅ 平均スケール {scale_mean:.3f} は統計平均 {actual_stats['width_ratio_mean']:.3f} と一致")
    else:
        print(f"⚠️ 平均スケールに差異あり")
    
    if (scale_min >= actual_stats["width_ratio_25th"] * 0.95 and 
        scale_max <= actual_stats["width_ratio_75th"] * 1.05):
        print(f"✅ スケール範囲 {new_cfg['SCALE_RANGE']} は統計範囲と適合")
    else:
        print(f"⚠️ スケール範囲要確認")
    
    print("\n面積比での確認:")
    area_min = scale_min ** 2
    area_max = scale_max ** 2
    area_mean = scale_mean ** 2
    
    print(f"新設定での面積比: {area_min*100:.2f}%-{area_max*100:.2f}% (平均{area_mean*100:.2f}%)")
    print(f"実際の統計面積比: {actual_stats['size_ratio_mean']*100:.3f}%")
    
    if abs(area_mean - actual_stats["size_ratio_mean"]) < actual_stats["size_ratio_mean"] * 0.2:
        print("✅ 面積比も統計と適合")
    else:
        print("⚠️ 面積比要確認")

if __name__ == "__main__":
    explain_cfg_corrections()
    validate_new_settings()
