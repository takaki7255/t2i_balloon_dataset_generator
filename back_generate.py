# back_generate.py
"""
漫画の背景画像を生成するスクリプト。
OpenAI API を使用して、指定されたプロンプトに基づいて画像を生成
"""
from openai import OpenAI
import base64
import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(".env")
api_key = os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

def get_next_filename(directory: str, prefix: str = "", extension: str = ".png") -> str:
    """次のナンバリングファイル名を取得"""
    os.makedirs(directory, exist_ok=True)
    
    existing_files = []
    for file in os.listdir(directory):
        if file.endswith(extension):
            # ファイル名から数字部分を抽出
            stem = Path(file).stem
            if stem.replace(prefix, "").isdigit():
                num = int(stem.replace(prefix, ""))
                existing_files.append(num)
    
    # 次の番号を決定
    next_num = 1 if not existing_files else max(existing_files) + 1
    filename = f"{prefix}{next_num:03d}{extension}"
    
    return os.path.join(directory, filename)

# 片側ページのレイアウトを指定する場合
prompt = """
漫画の背景画像を生成してください
人物やオノマトペ，吹き出しは配置されていない状態で、背景は白色です。
コマ割りは適度に単調にならないように横長のコマや縦長のコマが配置され，片側ページのレイアウトでお願いします。
コマ内には建物や風景，単調な模様，無地背景，グラデーション，物体など簡易的な背景を設置してください
コマはページ内に網羅的に設置してください
"""

# 見開きページの横長レイアウトを指定する場合
# prompt = """
# 漫画の背景画像を生成してください
# 吹き出しは配置されていない状態で、背景は白色です。
# コマ割りは適度に配置され，見開きページの横長のレイアウトでお願いします。
# 2ページで構成され、各ページには漫画のコマが配置されています。
# """

result = client.images.generate(
    model="gpt-image-1",
    prompt=prompt
)

image_base64 = result.data[0].b64_json
image_bytes = base64.b64decode(image_base64)

# 次のナンバリングファイル名を取得
output_path = get_next_filename("generated_backs")

# 画像を保存
with open(output_path, "wb") as f:
    f.write(image_bytes)

print(f"背景画像を保存しました: {output_path}")