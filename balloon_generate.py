from openai import OpenAI
import base64
import os
from dotenv import load_dotenv

load_dotenv(".env")
api_key = os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

"""
参考：吹き出しの形状と用途の一覧
| 区分      | 形状特徴        | 主な用途        |
| ------- | ----------- | ----------- |
| 丸／楕円型   | シンプルな輪郭＋尾   | 通常会話        |
| 矩形型     | 四角（尾なし・矩形尾） | ナレーション、ロボ声等 |
| ギザギザ型   | 爆発輪郭        | 叫び声・激しい感情   |
| 小さなトゲ   | 軽いギザギザ輪郭    | 怒り・苛立ち      |
| 波線型     | 波打つ輪郭       | 弱い声・恐怖      |
| 点線輪郭    | ドットで囲む      | ささやき・静かな話   |
| 雲型＋粒尾   | 雲状輪郭＋玉状尾    | 思考・心の声      |
| 放射状／爆尾  | 尾が放射状・爆発状   | 放送・機械の声     |
| Icicle型 | 下部が氷柱状      | 冷たさ・敵意      |
| モンスター型  | ドリップ装飾      | クリーチャーの声    |
| カラー型    | 背景吹き出し色     | 感情・キャラ特性強調  |
"""

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