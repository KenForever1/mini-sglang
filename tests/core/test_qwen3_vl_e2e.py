"""Quick smoke test for Qwen3-VL image understanding via OpenAI SDK."""

from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:1919/v1", api_key="none")

# Test 1: Pure text (sanity check)
print("=== Test 1: Pure text ===")
resp = client.chat.completions.create(
    model="Qwen3-VL-2B-Instruct",
    messages=[{"role": "user", "content": "1+1等于几？只回答数字"}],
    max_tokens=16,
    temperature=0,
    stream=True,
)
for chunk in resp:
    print(chunk.choices[0].delta.content or "", end="", flush=True)
print("\n")

# Test 2: Image URL + text
print("=== Test 2: Image URL ===")
resp = client.chat.completions.create(
    model="Qwen3-VL-2B-Instruct",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
                    },
                },
                {"type": "text", "text": "请描述这张图片的内容"},
            ],
        }
    ],
    max_tokens=256,
    temperature=0,
    stream=True,
)
for chunk in resp:
    print(chunk.choices[0].delta.content or "", end="", flush=True)
print("\n")

print("=== All tests done ===")


# === Test 1: Pure text ===
# 2

# === Test 2: Image URL ===
# 这是一张充满温馨与宁静氛围的户外照片，拍摄于一个阳光明媚的海滩。

# 画面中，一位年轻女性和一只金毛犬正坐在沙滩上，享受着美好的时光。女性留着深色长发，身穿一件蓝白格子衬衫和深色裤子，她面带微笑，侧身对着镜头，目光温柔地注视着身旁的狗狗。她的右手正与狗狗的前爪轻轻相触，形成一个友好的“击掌”动作，这个动作充满了亲密和快乐。狗狗也正坐着，用它那温暖的金色毛发和友好的姿态回应着主人，它身上佩戴着一个带有彩色爪印图案的狗项圈，显得十分可爱。

# 他们所处的环境是广阔的沙滩，沙子细腻，被阳光照得泛着柔和的光泽。背景是平静的海洋，海浪在远处轻轻拍打着海岸，海天相接处的天空呈现出温暖的橙黄色调，表明这是日出或日落时分。阳光从画面的右侧斜射过来，为整个场景镀上了一层金色的光辉，营造出一种温暖、浪漫的氛围。

# 整张照片构图和谐，色彩温暖，捕捉到了人与宠物

# === All tests done ===