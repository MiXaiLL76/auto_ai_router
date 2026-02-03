#!/usr/bin/env python3

import requests

API_KEY = "sk-your-master-key-here"
BASE_URL = "http://localhost:8080/v1"
MODEL = "gemini-2.5-flash"
VIDEO_URL = "https://storage.yandexcloud.net/ai-roman/1.mp4"

prompt = (
    "Опиши видео.\n"
    "Обязательно выпиши текст, который виден в кадре (если есть), дословно."
)

payload = {
    "model": MODEL,
    "temperature": 0.2,
    "messages": [{
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "file", "file": {"file_id": VIDEO_URL, "format": "video/mp4"}},
        ],
    }],
}

headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
r = requests.post(f"{BASE_URL}/chat/completions", headers=headers, json=payload, timeout=180)

print("HTTP", r.status_code)
r.raise_for_status()
print(r.json()["choices"][0]["message"]["content"])
