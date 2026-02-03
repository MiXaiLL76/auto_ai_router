#!/usr/bin/env python3
"""
Test script for image generation using OpenAI Python library
Tests the /v1/images/generations endpoint through the proxy
"""

import base64
import sys
from pathlib import Path

from openai import OpenAI

# Initialize OpenAI client pointing to local proxy
client = OpenAI(
    api_key="sk-your-master-key-here",
    base_url="http://localhost:8080/v1",
)

# Detailed prompt for image generation
prompt = """Создай выразительное превью-изображение для статьи.

Контекст статьи:
Тема: Разговоры ни о чем: как пустая коммуникация съедает время и энергию в бизнесе
Тип контента: informational

Визуальный подход: Дискуссия / обмен мнениями
Визуальная стратегия: realistic_scene
Уровень живости: high

Сцена и образ:
Два коллеги в офисе ведут оживлённый, но абсолютно бессмысленный разговор, стоя у огромного, пустого аквариума. Их жесты активны, лица выражают вовлечённость, но в аквариуме нет ни воды, ни рыб — только несколько декоративных камней на дне. Свет из окна падает на аквариум, подчёркивая его пустоту и создавая отражения на лицах людей. В фокусе — момент взаимодействия и иллюзия содержательного диалога.

Требования к стилю:
Реалистичная фотография, естественное освещение, глубина резкости, акцент на лицах и жестах людей на фоне символичного пустого объекта.

Настроение:
Ироничный, задумчивый, с лёгкой долей абсурда. Ощущение «подсмотренного момента», который заставляет зрителя задуматься о сути происходящего.

Ограничения (строго соблюдать):
Без текста, букв и цифр. Без логотипов и брендов. Без узнаваемых интерфейсов и скриншотов. Изображение должно выглядеть как живая обложка статьи, а не как иконка или презентация. Люди должны присутствовать в кадре (режим with_people). Избегать буквального показа «ничего» через пустые стены или экраны. Фокус на взаимодействии между людьми, а не на пустом объекте

Общие требования:
- Это обложка статьи для сайта, а не иллюстрация для презентации
- Фокус на сцене, эмоции или визуальной истории
- Один главный объект или персонаж
- Эффект живого, реального момента
- Качество: журнальный уровень, высокая детализация

Правила по уровню живости (high):
Допускаются небольшие несовершенства кадра.
Неидеальная поза, живой взгляд.
Эффект «подсмотренного момента».

Правила по стратегии (realistic_scene):
Жизненная или рабочая сцена.
Реалистичный свет и окружение.

Правила по типу контента (informational):
Человек в процессе изучения или применения знаний.
Момент понимания или открытия.
Практический контекст.

Дополнительно:
- Без текста и символов
- Без логотипов и брендов
- Подходит для обложки статьи
- Избегай композиций "рука тянется к продукту" или "рука трогает продукт" — это стоковый штамп"""

try:
    print("Sending image generation request...")
    print("Model: gemini-2.5-flash-image")
    print("Size: 1792x1024")
    print()

    response = client.images.generate(
        model="gemini-2.5-flash-image",
        prompt=prompt,
        size="1792x1024",
        quality="standard",
        n=1,
    )

    print("Response received!")
    print(f"Created: {response.created}")
    print(f"Number of images: {len(response.data)}")
    print()

    # Save images
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    for i, image_data in enumerate(response.data):
        try:
            if image_data.b64_json:
                # Decode base64 and save
                print(f"Processing image {i}...")
                print(f"  b64_json type: {type(image_data.b64_json)}")
                print(f"  b64_json length: {len(image_data.b64_json) if hasattr(image_data.b64_json, '__len__') else 'N/A'}")

                if isinstance(image_data.b64_json, str):
                    image_bytes = base64.b64decode(image_data.b64_json)
                else:
                    image_bytes = image_data.b64_json

                output_path = output_dir / f"generated_image_{i}.png"
                with open(str(output_path), "wb") as f:
                    f.write(image_bytes)
                print(f"✓ Saved image {i} to {output_path}")
                print(f"  File size: {len(image_bytes)} bytes")
            elif image_data.url:
                print(f"✓ Image {i} URL: {image_data.url}")
        except Exception as e:
            print(f"Error saving image {i}: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()

    print()
    print("Image generation completed successfully!")

except Exception as e:
    print(f"Error: {e}", file=sys.stderr)
    sys.exit(1)
