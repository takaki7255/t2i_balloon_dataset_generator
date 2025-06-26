# back_generate.py
"""
漫画の背景画像を生成するスクリプト。
OpenAI API を使用して、指定されたプロンプトに基づいて画像を生成
"""
from openai import OpenAI
import base64
import os
from dotenv import load_dotenv

load_dotenv(".env")
api_key = os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# 片側ページのレイアウトを指定する場合
prompt = """
漫画の背景画像を生成してください
吹き出しは配置されていない状態で、背景は白色です。
コマ割りは適度に配置され，片側ページのレイアウトでお願いします。
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

# Save the image to a file
with open("generated_image/output.png", "wb") as f:
    f.write(image_bytes)