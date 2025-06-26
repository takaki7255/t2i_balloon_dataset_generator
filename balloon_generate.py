from openai import OpenAI
import base64
import os
from dotenv import load_dotenv

load_dotenv(".env")
api_key = os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

prompt = """
漫画の吹き出し
白色無地背景
吹き出し内には日本語テキストを挿入し，テキストは吹き出し内に収まるようにしてください
テキストや吹き出しの形・色は以下の条件に従ってください
吹き出し内の色が白色の場合はテキスト色を黒．吹き出し内の色が黒色の場合はテキスト色を白にしてください
吹き出しの形：形状の説明
吹き出しの色：白/黒
テキスト：吹き出し内テキストの指定
先端部分の有無：有/無
"""

result = client.images.generate(
    model="gpt-image-1",
    prompt=prompt
)

image_base64 = result.data[0].b64_json
image_bytes = base64.b64decode(image_base64)

# Save the image to a file
with open("generated_image/output.png", "wb") as f:
    f.write(image_bytes)