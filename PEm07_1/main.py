#!/usr/bin/env python3
"""
CLI приложение для анализа фотографий блюд с использованием YandexGPT Vision API.
Определяет название блюда, ингредиенты, тип кухни и уровень уверенности.
"""

import os
import sys
import json
import base64
import argparse
from pathlib import Path
from typing import Dict, Any

import requests
from dotenv import load_dotenv


def load_environment() -> tuple[str, str]:
    """Загружает переменные окружения из .env файла."""
    load_dotenv()
    
    api_key = os.getenv('YANDEX_GPT_API_KEY')
    folder_id = os.getenv('YANDEX_GPT_FOLDER_ID')
    
    if not api_key:
        print("Ошибка: YANDEX_GPT_API_KEY не найден в .env файле", file=sys.stderr)
        sys.exit(1)
    
    if not folder_id:
        print("Ошибка: YANDEX_GPT_FOLDER_ID не найден в .env файле", file=sys.stderr)
        sys.exit(1)
    
    return api_key, folder_id


def encode_image(image_path: Path) -> str:
    """Кодирует изображение в base64."""
    try:
        with open(image_path, 'rb') as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        print(f"Ошибка: Файл {image_path} не найден", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Ошибка при чтении файла: {e}", file=sys.stderr)
        sys.exit(1)


def recognize_dish_with_vision(image_path: Path, api_key: str, folder_id: str) -> Dict[str, Any] | None:
    """
    Распознает блюдо на изображении с помощью Yandex Vision API (модель food).
    
    Args:
        image_path: Путь к изображению
        api_key: API ключ Yandex Cloud
        folder_id: ID каталога в Yandex Cloud
    
    Returns:
        Словарь с dish_name и confidence или None, если не удалось распознать
    """
    # Кодируем изображение в base64
    image_base64 = encode_image(image_path)
    
    # Формируем запрос к Vision API
    url = "https://vision.api.cloud.yandex.net/vision/v1/batchAnalyze"
    
    headers = {
        "Authorization": f"Api-Key {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "folderId": folder_id,
        "analyzeSpecs": [
            {
                "content": image_base64,
                "features": [
                    {
                        "type": "CLASSIFICATION",
                         "classificationConfig": {
                             "model": "all"  # модель для распознавания еды
                         }
                    }
                ]
            }
        ]
    }
    
    try:
        print("Отправка запроса в Vision API (модель food)...", file=sys.stderr)
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        
        if response.status_code != 200:
            print(f"Ошибка Vision API: {response.status_code}, {response.text}", file=sys.stderr)
            return None
        
        result = response.json()
        
        # Отладочная информация: выводим структуру ответа
        print(f"Полный ответ Vision API: {json.dumps(result, indent=2, ensure_ascii=False)}", file=sys.stderr)
        
        # Извлекаем метки из ответа согласно структуре batchAnalyze
        # Структура может быть: result['results'][0]['results'][0]['classification']['properties']
        if "results" in result and len(result["results"]) > 0:
            first_result = result["results"][0]
            
            properties = None
            
            # Вариант 1: batchAnalyze структура - results[0].results[].classification.properties
            if "results" in first_result:
                for analyze_result in first_result["results"]:
                    if "classification" in analyze_result:
                        classification = analyze_result["classification"]
                        if "properties" in classification:
                            properties = classification["properties"]
                            print(f"Найдены properties в classification: {len(properties)} меток", file=sys.stderr)
                            break
            
            # Вариант 2: batchClassify структура - results[0].analyzeResults[0].properties
            if not properties and "analyzeResults" in first_result:
                if len(first_result["analyzeResults"]) > 0:
                    classification_result = first_result["analyzeResults"][0]
                    if "properties" in classification_result:
                        properties = classification_result["properties"]
                        print(f"Найдены properties в analyzeResults: {len(properties)} меток", file=sys.stderr)
            
            if properties and len(properties) > 0:
                # Находим метку с максимальным score (уверенностью)
                best_label = max(properties, key=lambda x: float(x.get("score", 0)))
                dish_name = best_label.get("name", "")
                confidence = float(best_label.get("score", 0))
                
                print(f"Vision API определил блюдо: {dish_name} (уверенность: {confidence:.2f})", file=sys.stderr)
                
                return {
                    "dish_name": dish_name,
                    "confidence": confidence
                }
            else:
                print("Не найдены properties в ответе Vision API", file=sys.stderr)
        else:
            print("Ответ Vision API не содержит results или results пуст", file=sys.stderr)
        
        print("Vision API не вернул результатов распознавания", file=sys.stderr)
        return None
        
    except Exception as e:
        print(f"Ошибка при запросе к Vision API: {e}", file=sys.stderr)
        return None


def get_ingredients_and_cuisine(dish_name: str, api_key: str, folder_id: str) -> Dict[str, Any]:
    """
    Определяет ингредиенты и тип кухни на основе названия блюда с помощью YandexGPT.
    
    Args:
        dish_name: Название блюда (на английском, от Vision API)
        api_key: API ключ YandexGPT
        folder_id: ID каталога в Yandex Cloud
    
    Returns:
        Словарь с ingredients и cuisine
    """
    url = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
    
    headers = {
        "Authorization": f"Api-Key {api_key}",
        "Content-Type": "application/json"
    }
    
    model_uri = f"gpt://{folder_id}/yandexgpt/latest"
    
    prompt = f"""Ты - эксперт по кулинарии.

Yandex Vision API распознал на изображении блюдо: "{dish_name}".

На основе названия блюда определи:
1. Полное название блюда на русском языке (если нужно, переведи или уточни)
2. Список основных ингредиентов, которые обычно используются в этом блюде
3. Тип кухни (например: итальянская, русская, японская, французская, китайская, мексиканская и т.д.)

Верни ответ строго в формате JSON:
{{
  "dish_name": "полное название блюда на русском",
  "ingredients": ["ингредиент1", "ингредиент2", ...],
  "cuisine": "тип кухни"
}}

Важно: 
- Верни только валидный JSON, без дополнительных комментариев или текста
- Если название блюда на английском, переведи его на русский
- Укажи реальные ингредиенты, которые обычно используются в этом блюде"""
    
    payload = {
        "modelUri": model_uri,
        "completionOptions": {
            "stream": False,
            "temperature": 0.2,
            "maxTokens": 500
        },
        "messages": [
            {
                "role": "user",
                "text": prompt
            }
        ]
    }
    
    try:
        print("Запрос к YandexGPT для определения ингредиентов и типа кухни...", file=sys.stderr)
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        
        if "result" in result and "alternatives" in result["result"]:
            if len(result["result"]["alternatives"]) > 0:
                text_response = result["result"]["alternatives"][0]["message"]["text"]
                return _parse_gpt_response(text_response)
        
        print("Ошибка: Неожиданный формат ответа от YandexGPT", file=sys.stderr)
        return {"dish_name": dish_name, "ingredients": [], "cuisine": "неизвестна"}
        
    except Exception as e:
        print(f"Ошибка при запросе к YandexGPT: {e}", file=sys.stderr)
        return {"dish_name": dish_name, "ingredients": [], "cuisine": "неизвестна"}


def analyze_dish_image(image_path: Path, api_key: str, folder_id: str) -> Dict[str, Any]:
    """
    Анализирует изображение блюда: сначала Vision API для распознавания,
    затем YandexGPT для определения ингредиентов и типа кухни.
    
    Args:
        image_path: Путь к изображению
        api_key: API ключ Yandex Cloud
        folder_id: ID каталога в Yandex Cloud
    
    Returns:
        Словарь с результатами анализа
    """
    # Шаг 1: Распознаем блюдо с помощью Vision API
    vision_result = recognize_dish_with_vision(image_path, api_key, folder_id)
    
    if not vision_result or not vision_result.get("dish_name"):
        # Если Vision API не смог распознать блюдо
        return {
            "dish_name": None,
            "ingredients": [],
            "cuisine": None,
            "confidence": 0.0
        }
    
    dish_name_en = vision_result["dish_name"]
    confidence = vision_result["confidence"]
    
    # Шаг 2: Определяем ингредиенты и тип кухни с помощью YandexGPT
    gpt_result = get_ingredients_and_cuisine(dish_name_en, api_key, folder_id)
    
    # Объединяем результаты
    return {
        "dish_name": gpt_result.get("dish_name", dish_name_en),
        "ingredients": gpt_result.get("ingredients", []),
        "cuisine": gpt_result.get("cuisine", "неизвестна"),
        "confidence": confidence
    }




def _parse_gpt_response(text_response: str) -> Dict[str, Any]:
    """Парсит ответ от YandexGPT в JSON."""
    try:
        # Убираем возможные markdown форматирование
        text_response = text_response.strip()
        if text_response.startswith("```json"):
            text_response = text_response[7:]
        if text_response.startswith("```"):
            text_response = text_response[3:]
        if text_response.endswith("```"):
            text_response = text_response[:-3]
        text_response = text_response.strip()
        
        parsed_result = json.loads(text_response)
        return parsed_result
    except json.JSONDecodeError as e:
        print(f"Ошибка парсинга JSON ответа: {e}", file=sys.stderr)
        print(f"Ответ модели: {text_response}", file=sys.stderr)
        sys.exit(1)




def main():
    """Основная функция CLI приложения."""
    parser = argparse.ArgumentParser(
        description='Анализ фотографии блюда с определением названия, ингредиентов и типа кухни',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python main.py dish.jpg
  python main.py path/to/image.png
        """
    )
    parser.add_argument(
        'image_path',
        type=str,
        help='Путь к изображению блюда'
    )
    parser.add_argument(
        '--output',
        '-o',
        type=str,
        help='Путь для сохранения результата в JSON файл (опционально)'
    )
    
    args = parser.parse_args()
    
    # Проверяем существование файла
    image_path = Path(args.image_path)
    if not image_path.exists():
        print(f"Ошибка: Файл {image_path} не существует", file=sys.stderr)
        sys.exit(1)
    
    # Загружаем переменные окружения
    api_key, folder_id = load_environment()
    
    # Анализируем изображение
    result = analyze_dish_image(image_path, api_key, folder_id)
    
    # Форматируем результат в JSON
    output_json = json.dumps(result, ensure_ascii=False, indent=2)
    
    # Выводим результат
    print(output_json)
    
    # Сохраняем в файл, если указан
    if args.output:
        output_path = Path(args.output)
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(output_json)
            print(f"\nРезультат сохранен в {output_path}", file=sys.stderr)
        except Exception as e:
            print(f"Ошибка при сохранении файла: {e}", file=sys.stderr)
            sys.exit(1)


if __name__ == '__main__':
    main()

