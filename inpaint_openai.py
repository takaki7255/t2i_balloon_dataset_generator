from openai import OpenAI
import base64
import os
import requests
from dotenv import load_dotenv
from pathlib import Path
from PIL import Image

load_dotenv(".env")
api_key = os.environ.get("OPENAI_API_KEY")

# APIキーの確認
if not api_key:
    print("エラー: OPENAI_API_KEYが設定されていません。")
    print("以下のいずれかの方法でAPIキーを設定してください:")
    print("1. .envファイルに OPENAI_API_KEY=your_api_key_here を追加")
    print("2. 環境変数として export OPENAI_API_KEY=your_api_key_here を実行")
    exit(1)

try:
    client = OpenAI(api_key=api_key)
    print("OpenAI APIクライアントを初期化しました。")
except Exception as e:
    print(f"OpenAI APIクライアントの初期化に失敗しました: {e}")
    exit(1)

def encode_image_to_base64(image_path: str) -> str:
    """画像ファイルをbase64にエンコード"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def convert_to_rgba(image_path: str, output_path: str):
    """画像をRGBA形式に変換"""
    with Image.open(image_path) as img:
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        img.save(output_path)
    return output_path

# ハイキューのパネル画像とマスクをRGBA形式に変換
print("画像を処理中...")
image_rgba_path = convert_to_rgba("haikyu_panel.png", "haikyu_panel_rgba.png")
mask_rgba_path = convert_to_rgba("haikyu_panel_mask.png", "haikyu_panel_mask_rgba.png")

print(f"RGBA変換完了: {image_rgba_path}, {mask_rgba_path}")

try:
    # DALL-E 3による画像編集（インペインティング）
    response = client.images.edit(
        image=open(image_rgba_path, "rb"),
        mask=open(mask_rgba_path, "rb"),
        prompt="マスクされた部分に適切な漫画の背景要素を生成してください。マスク領域の周辺情報から元の漫画スタイルを維持してください。",
        n=1,
        size="1024x1024"
    )
    
    # 生成された画像のURLを取得
    image_url = response.data[0].url
    print(f"生成された画像URL: {image_url}")
    
    # URLから画像をダウンロードして保存
    response = requests.get(image_url)
    if response.status_code == 200:
        output_filename = "haikyu_inpainted.png"
        with open(output_filename, "wb") as f:
            f.write(response.content)
        print(f"補完された画像を保存しました: {output_filename}")
    else:
        print("画像のダウンロードに失敗しました。")

except Exception as e:
    print(f"エラーが発生しました: {e}")
    print("代替手法を試行します...")
    
    # 代替手法: 画像生成API
    try:
        response = client.images.generate(
            model="dall-e-3",
            prompt="ハイキューの漫画スタイルで、バレーボールコートの背景を生成してください。",
            size="1024x1024",
            quality="standard",
            n=1,
        )
        
        image_url = response.data[0].url
        print(f"生成された画像URL: {image_url}")
        
        # URLから画像をダウンロード
        img_response = requests.get(image_url)
        if img_response.status_code == 200:
            output_filename = "haikyu_generated.png"
            with open(output_filename, "wb") as f:
                f.write(img_response.content)
            print(f"生成された画像を保存しました: {output_filename}")
        
    except Exception as e2:
        print(f"代替手法でもエラーが発生しました: {e2}")