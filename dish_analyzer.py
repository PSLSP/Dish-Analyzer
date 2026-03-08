#!/usr/bin/env python3
"""
CLI приложение для анализа изображений блюд с помощью OpenAI Vision API и GigaChat.
"""

import os
import sys
import json
import argparse
import base64
import time
import uuid
import urllib3
from pathlib import Path
from typing import Dict, Any, Optional

from dotenv import load_dotenv
from openai import OpenAI
import requests

# Отключаем предупреждения о небезопасных SSL запросах, если проверка отключена
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Кэш для Access token GigaChat
_gigachat_access_token = None
_gigachat_token_expires_at = 0
GIGACHAT_TOKEN_LIFETIME = 30 * 60  # 30 минут в секундах


def load_image_as_base64(image_path: str) -> str:
    """Загружает изображение и конвертирует в base64."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        print(f"Ошибка: Файл '{image_path}' не найден.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Ошибка при чтении файла: {e}", file=sys.stderr)
        sys.exit(1)


def analyze_dish(
    image_path: str,
    api_key: str,
    model: str = "gpt-4o",
    max_tokens: int = 500,
    temperature: float = 0.3
) -> Dict[str, Any]:
    """
    Анализирует изображение блюда с помощью OpenAI Vision API.
    
    Args:
        image_path: Путь к изображению
        api_key: OpenAI API ключ
        model: Модель OpenAI (по умолчанию "gpt-4o")
        max_tokens: Максимальное количество токенов (по умолчанию 500)
        temperature: Температура модели (по умолчанию 0.3)
    
    Returns:
        Словарь с информацией о блюде
    """
    # Загружаем изображение
    base64_image = load_image_as_base64(image_path)
    
    # Определяем MIME тип
    image_ext = Path(image_path).suffix.lower()
    mime_types = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.gif': 'image/gif',
        '.webp': 'image/webp'
    }
    mime_type = mime_types.get(image_ext, 'image/jpeg')
    
    # Инициализируем клиент OpenAI
    client = OpenAI(api_key=api_key)
    
    # Промпт для структурированного вывода
    prompt = """Проанализируй это изображение блюда и предоставь информацию в следующем JSON формате:
{
  "dish_name": "название блюда",
  "ingredients": ["ингредиент1", "ингредиент2", ...],
  "cuisine": "тип кухни",
  "confidence": 0.0-1.0
}

Важно:
- dish_name: точное название блюда на русском языке
- ingredients: список предполагаемых ингредиентов на русском языке
- cuisine: тип кухни на русском языке (например, "итальянская", "японская", "русская")
- confidence: уровень уверенности от 0.0 до 1.0

Верни ТОЛЬКО валидный JSON, без дополнительных комментариев или текста."""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        # Извлекаем ответ
        content = response.choices[0].message.content.strip()
        
        # Пытаемся извлечь JSON из ответа
        # Иногда модель может обернуть JSON в markdown код блоки
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        
        # Парсим JSON
        try:
            result = json.loads(content)
            
            # Валидация структуры
            required_fields = ["dish_name", "ingredients", "cuisine", "confidence"]
            for field in required_fields:
                if field not in result:
                    raise ValueError(f"Отсутствует обязательное поле: {field}")
            
            # Проверяем типы
            if not isinstance(result["dish_name"], str):
                raise ValueError("dish_name должен быть строкой")
            if not isinstance(result["ingredients"], list):
                raise ValueError("ingredients должен быть списком")
            if not isinstance(result["cuisine"], str):
                raise ValueError("cuisine должен быть строкой")
            if not isinstance(result["confidence"], (int, float)) or not (0 <= result["confidence"] <= 1):
                raise ValueError("confidence должен быть числом от 0.0 до 1.0")
            
            return result
            
        except json.JSONDecodeError as e:
            print(f"Ошибка: Не удалось распарсить JSON ответ от API: {e}", file=sys.stderr)
            print(f"Ответ API: {content}", file=sys.stderr)
            sys.exit(1)
        except ValueError as e:
            print(f"Ошибка валидации: {e}", file=sys.stderr)
            print(f"Ответ API: {content}", file=sys.stderr)
            sys.exit(1)
            
    except Exception as e:
        print(f"Ошибка при обращении к OpenAI API: {e}", file=sys.stderr)
        sys.exit(1)


def get_gigachat_access_token(api_key: str, scope: str, token_url: str, verify_ssl: bool = True) -> str:
    """
    Получает Access token для авторизации запросов к GigaChat API.
    Токен кэшируется и обновляется при необходимости (действует 30 минут).
    
    Args:
        api_key: GigaChat API ключ (Base64 encoded)
        scope: Scope для авторизации
        token_url: URL для получения токена
        verify_ssl: Проверять ли SSL сертификат
    
    Returns:
        Access token
    """
    global _gigachat_access_token, _gigachat_token_expires_at
    
    # Проверяем, не истек ли токен (оставляем запас в 1 минуту)
    current_time = time.time()
    if _gigachat_access_token and current_time < (_gigachat_token_expires_at - 60):
        return _gigachat_access_token
    
    # Получаем новый токен
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded',
        'Accept': 'application/json',
        'RqUID': str(uuid.uuid4()),
        'Authorization': f'Basic {api_key}'
    }
    
    payload = {
        'scope': scope
    }
    
    try:
        response = requests.post(
            token_url,
            headers=headers,
            data=payload,
            timeout=30,
            verify=verify_ssl
        )
        
        if response.status_code == 401:
            error_detail = ""
            try:
                error_data = response.json()
                error_detail = f" Детали: {error_data}"
            except:
                error_detail = f" Ответ сервера: {response.text[:200]}"
            raise Exception(
                f"Ошибка авторизации (401). Проверьте правильность GIGACHAT_API_KEY и GIGACHAT_SCOPE.{error_detail}"
            )
        
        response.raise_for_status()
        
        data = response.json()
        access_token = data.get('access_token')
        
        if not access_token:
            raise ValueError("Access token не найден в ответе API")
        
        # Сохраняем время истечения токена
        _gigachat_access_token = access_token
        _gigachat_token_expires_at = current_time + GIGACHAT_TOKEN_LIFETIME
        
        return access_token
        
    except requests.exceptions.RequestException as e:
        raise Exception(f"Ошибка при получении Access token: {str(e)}")


def format_recipe_as_text(recipe_data: Dict[str, Any]) -> str:
    """
    Форматирует рецепт в читаемый текстовый формат.
    
    Args:
        recipe_data: Данные рецепта от GigaChat
    
    Returns:
        Отформатированная строка с рецептом
    """
    if not recipe_data:
        return ""
    
    output = []
    
    # Заголовок рецепта
    recipe = recipe_data.get("recipe", {})
    if recipe:
        output.append("=" * 60)
        output.append(f"  {recipe.get('name', 'Рецепт')}")
        output.append("=" * 60)
        output.append("")
        
        # Информация о рецепте
        if recipe.get("cuisine"):
            output.append(f"Кухня: {recipe['cuisine']}")
        if recipe.get("servings"):
            output.append(f"Порций: {recipe['servings']}")
        if recipe.get("prep_time"):
            output.append(f"Время подготовки: {recipe['prep_time']}")
        if recipe.get("cook_time"):
            output.append(f"Время приготовления: {recipe['cook_time']}")
        if recipe.get("total_time"):
            output.append(f"Общее время: {recipe['total_time']}")
        output.append("")
        
        # Ингредиенты
        ingredients = recipe.get("ingredients", [])
        if ingredients:
            output.append("ИНГРЕДИЕНТЫ:")
            output.append("-" * 60)
            for ing in ingredients:
                name = ing.get("name", "")
                amount = ing.get("amount", "")
                notes = ing.get("notes", "")
                if notes:
                    output.append(f"  • {name}: {amount} ({notes})")
                else:
                    output.append(f"  • {name}: {amount}")
            output.append("")
    
    # Этапы приготовления
    cooking_steps = recipe_data.get("cooking_steps", [])
    if cooking_steps:
        output.append("ЭТАПЫ ПРИГОТОВЛЕНИЯ:")
        output.append("-" * 60)
        for step in cooking_steps:
            step_num = step.get("step_number", "")
            description = step.get("description", "")
            duration = step.get("duration", "")
            if duration:
                output.append(f"\nШаг {step_num}: {description}")
                output.append(f"  Время: {duration}")
            else:
                output.append(f"\nШаг {step_num}: {description}")
        output.append("")
    
    return "\n".join(output)


def format_nutrition_table(nutrition_data: Dict[str, Any]) -> str:
    """
    Форматирует КБЖУ и ХЕ в виде таблицы.
    
    Args:
        nutrition_data: Данные о питательной ценности
    
    Returns:
        Отформатированная таблица
    """
    if not nutrition_data:
        return ""
    
    output = []
    output.append("ПИТАТЕЛЬНАЯ ЦЕННОСТЬ")
    output.append("=" * 60)
    
    # Заголовок таблицы
    header = f"{'Параметр':<20} {'На порцию':<15} {'На 100г':<15}"
    output.append(header)
    output.append("-" * 60)
    
    per_serving = nutrition_data.get("per_serving", {})
    per_100g = nutrition_data.get("per_100g", {})
    
    # Калории
    calories_serving = per_serving.get("calories", 0)
    calories_100g = per_100g.get("calories", 0)
    output.append(f"{'Калории (ккал)':<20} {calories_serving:<15} {calories_100g:<15}")
    
    # Белки
    protein_serving = per_serving.get("protein", 0)
    protein_100g = per_100g.get("protein", 0)
    output.append(f"{'Белки (г)':<20} {protein_serving:<15} {protein_100g:<15}")
    
    # Жиры
    fat_serving = per_serving.get("fat", 0)
    fat_100g = per_100g.get("fat", 0)
    output.append(f"{'Жиры (г)':<20} {fat_serving:<15} {fat_100g:<15}")
    
    # Углеводы
    carbs_serving = per_serving.get("carbs", 0)
    carbs_100g = per_100g.get("carbs", 0)
    output.append(f"{'Углеводы (г)':<20} {carbs_serving:<15} {carbs_100g:<15}")
    
    # Хлебные единицы
    bread_units_serving = per_serving.get("bread_units", 0)
    bread_units_100g = per_100g.get("bread_units", 0)
    output.append(f"{'Хлебные единицы (ХЕ)':<20} {bread_units_serving:<15} {bread_units_100g:<15}")
    
    output.append("=" * 60)
    output.append("")
    
    return "\n".join(output)


def generate_image_with_openai(
    prompt: str,
    api_key: str,
    model: str = "dall-e-2",
    size: str = "1024x1024",
    quality: str = "standard",
    n: int = 1,
    response_format: str = "url"
) -> Dict[str, Any]:
    """
    Генерирует изображение с помощью OpenAI DALL-E API.
    
    Args:
        prompt: Промпт для генерации изображения
        api_key: OpenAI API ключ
        model: Модель для генерации (dall-e-2 или dall-e-3)
        size: Размер изображения (для dall-e-2: 256x256, 512x512, 1024x1024;
              для dall-e-3: 1024x1024, 1792x1024, 1024x1792)
        quality: Качество изображения (только для dall-e-3: standard или hd)
        n: Количество изображений (для dall-e-3 всегда 1)
        response_format: Формат ответа (url или b64_json)
    
    Returns:
        Словарь с информацией о сгенерированном изображении
    """
    client = OpenAI(api_key=api_key)
    
    try:
        # Для dall-e-3 параметр n всегда равен 1
        if model == "dall-e-3":
            params = {
                "model": model,
                "prompt": prompt,
                "size": size,
                "quality": quality,
                "n": 1,
                "response_format": response_format
            }
        else:
            # Для dall-e-2
            params = {
                "model": model,
                "prompt": prompt,
                "size": size,
                "n": n,
                "response_format": response_format
            }
        
        response = client.images.generate(**params)
        
        # Извлекаем информацию об изображении
        image_data = response.data[0]
        
        result = {
            "url": image_data.url if hasattr(image_data, 'url') else None,
            "b64_json": image_data.b64_json if hasattr(image_data, 'b64_json') else None
        }
        
        return result
        
    except Exception as e:
        raise Exception(f"Ошибка при генерации изображения: {str(e)}")


def download_image_from_url(url: str, output_path: str) -> None:
    """
    Скачивает изображение по URL и сохраняет в файл.
    
    Args:
        url: URL изображения
        output_path: Путь для сохранения файла
    """
    try:
        # Создаем директорию, если её нет
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        with open(output_path, 'wb') as f:
            f.write(response.content)
            
    except Exception as e:
        raise Exception(f"Ошибка при скачивании изображения: {str(e)}")


def save_image_from_base64(b64_json: str, output_path: str) -> None:
    """
    Сохраняет изображение из base64 в файл.
    
    Args:
        b64_json: Base64 строка изображения
        output_path: Путь для сохранения файла
    """
    try:
        # Создаем директорию, если её нет
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        image_data = base64.b64decode(b64_json)
        with open(output_path, 'wb') as f:
            f.write(image_data)
            
    except Exception as e:
        raise Exception(f"Ошибка при сохранении изображения: {str(e)}")


def format_output_text(result: Dict[str, Any]) -> str:
    """
    Форматирует весь результат в текстовый формат.
    
    Args:
        result: Полный результат анализа
    
    Returns:
        Отформатированный текст
    """
    output = []
    
    # Информация о блюде
    dish_analysis = result.get("dish_analysis", {})
    if dish_analysis:
        output.append("=" * 60)
        output.append("АНАЛИЗ БЛЮДА")
        output.append("=" * 60)
        output.append(f"Название: {dish_analysis.get('dish_name', 'Неизвестно')}")
        output.append(f"Кухня: {dish_analysis.get('cuisine', 'Неизвестно')}")
        output.append(f"Уверенность: {dish_analysis.get('confidence', 0):.2f}")
        ingredients = dish_analysis.get('ingredients', [])
        if ingredients:
            output.append(f"Ингредиенты: {', '.join(ingredients)}")
        output.append("")
    
    # Рецепт
    recipe_data = result.get("recipe")
    if recipe_data:
        output.append(format_recipe_as_text(recipe_data))
        
        # КБЖУ таблица
        nutrition = recipe_data.get("nutrition")
        if nutrition:
            output.append(format_nutrition_table(nutrition))
        
        # Промпт для изображения
        image_prompt = recipe_data.get("image_prompt")
        if image_prompt:
            output.append("ПРОМПТ ДЛЯ ГЕНЕРАЦИИ ИЗОБРАЖЕНИЯ:")
            output.append("-" * 60)
            output.append(image_prompt)
            output.append("")
        
        # Информация о сгенерированном изображении
        generated_image = result.get("generated_image")
        if generated_image:
            output.append("СГЕНЕРИРОВАННОЕ ИЗОБРАЖЕНИЕ:")
            output.append("-" * 60)
            if generated_image.get("file_path"):
                output.append(f"Изображение сохранено: {generated_image['file_path']}")
            if generated_image.get("model"):
                output.append(f"Модель: {generated_image['model']}")
            output.append("")
    else:
        if "error" in result:
            output.append(f"Ошибка при получении рецепта: {result['error']}")
    
    # Информация о сгенерированном изображении (если рецепт не получен, но изображение есть)
    generated_image = result.get("generated_image")
    if generated_image and not recipe_data:
        output.append("СГЕНЕРИРОВАННОЕ ИЗОБРАЖЕНИЕ:")
        output.append("-" * 60)
        if generated_image.get("file_path"):
            output.append(f"Изображение сохранено: {generated_image['file_path']}")
        output.append("")
    
    return "\n".join(output)


def generate_recipe_with_gigachat(
    dish_info: Dict[str, Any],
    api_key: str,
    scope: str,
    api_url: str,
    token_url: str,
    model: str = "GigaChat-2",
    max_tokens: int = 4000,
    temperature: float = 0.7,
    repetition_penalty: float = 1.1,
    verify_ssl: bool = True
) -> Dict[str, Any]:
    """
    Генерирует рецепт, этапы приготовления, КБЖУ и промпт для изображения с помощью GigaChat-2.
    
    Args:
        dish_info: Информация о блюде от OpenAI (dish_name, ingredients, cuisine, confidence)
        api_key: GigaChat API ключ
        scope: Scope для авторизации
        api_url: URL API GigaChat
        token_url: URL для получения токена
        model: Модель GigaChat (по умолчанию GigaChat-2)
        max_tokens: Максимальное количество токенов (по умолчанию 4000)
        temperature: Температура модели (по умолчанию 0.7)
        repetition_penalty: Штраф за повторения (по умолчанию 1.1)
        verify_ssl: Проверять ли SSL сертификат
    
    Returns:
        Словарь с рецептом, этапами, КБЖУ и промптом
    """
    # Получаем Access token
    try:
        access_token = get_gigachat_access_token(api_key, scope, token_url, verify_ssl)
    except Exception as e:
        raise Exception(f"Ошибка авторизации GigaChat: {str(e)}")
    
    # Формируем промпт для GigaChat
    prompt = f"""На основе следующей информации о блюде создай подробный рецепт:

Название блюда: {dish_info['dish_name']}
Ингредиенты: {', '.join(dish_info['ingredients'])}
Тип кухни: {dish_info['cuisine']}

Создай структурированный ответ в формате JSON со следующей структурой:

{{
  "recipe": {{
    "name": "название блюда",
    "cuisine": "тип кухни",
    "servings": количество порций (число),
    "prep_time": "время подготовки (например, '15 минут')",
    "cook_time": "время приготовления (например, '30 минут')",
    "total_time": "общее время (например, '45 минут')",
    "ingredients": [
      {{
        "name": "название ингредиента",
        "amount": "количество (например, '200 г' или '2 шт')",
        "notes": "дополнительные заметки (опционально)"
      }}
    ]
  }},
  "cooking_steps": [
    {{
      "step_number": номер шага (число),
      "description": "подробное описание шага",
      "duration": "время выполнения (например, '5 минут')"
    }}
  ],
  "nutrition": {{
    "per_serving": {{
      "calories": калории (число),
      "protein": белки в граммах (число),
      "fat": жиры в граммах (число),
      "carbs": углеводы в граммах (число),
      "bread_units": хлебные единицы (число, округлить до 1 знака после запятой)
    }},
    "per_100g": {{
      "calories": калории (число),
      "protein": белки в граммах (число),
      "fat": жиры в граммах (число),
      "carbs": углеводы в граммах (число),
      "bread_units": хлебные единицы (число, округлить до 1 знака после запятой)
    }}
  }},
  "image_prompt": "детальный промпт на английском языке для генерации фотореалистичного изображения этого блюда. Промпт должен быть подробным, описывать внешний вид, стиль подачи, освещение, композицию. Обязательно укажи, что это должна быть фотография (photograph, photo), а не рисунок или иллюстрация. Формат: описание блюда, фотография (photograph), реалистичный стиль, естественное освещение, фон, композиция"
}}

Важно:
- Все текстовые поля на русском языке, кроме image_prompt (он на английском)
- bread_units рассчитывается как углеводы / 12 (1 ХЕ = 12 г углеводов)
- Промпт для изображения должен быть детальным и профессиональным
- Верни ТОЛЬКО валидный JSON, без дополнительных комментариев или текста
- Не используй markdown код блоки, только чистый JSON"""

    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'Authorization': f'Bearer {access_token}'
    }
    
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "n": 1,
        "stream": False,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "repetition_penalty": repetition_penalty,
        "update_interval": 0
    }
    
    try:
        response = requests.post(
            api_url,
            headers=headers,
            json=payload,
            timeout=60,
            verify=verify_ssl
        )
        response.raise_for_status()
        
        data = response.json()
        
        # Извлекаем ответ из структуры ответа API
        if "choices" in data and len(data["choices"]) > 0:
            response_text = data["choices"][0]["message"]["content"].strip()
            
            # Пытаемся извлечь JSON из ответа
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            # Парсим JSON
            try:
                recipe_data = json.loads(response_text)
                
                # Валидация структуры
                required_sections = ["recipe", "cooking_steps", "nutrition", "image_prompt"]
                for section in required_sections:
                    if section not in recipe_data:
                        raise ValueError(f"Отсутствует обязательная секция: {section}")
                
                return recipe_data
                
            except json.JSONDecodeError as e:
                raise Exception(f"Не удалось распарсить JSON ответ от GigaChat: {e}. Ответ: {response_text[:500]}")
            except ValueError as e:
                raise Exception(f"Ошибка валидации ответа GigaChat: {e}")
        else:
            raise Exception("Не удалось получить ответ от GigaChat API")
            
    except requests.exceptions.RequestException as e:
        raise Exception(f"Ошибка при обращении к GigaChat API: {str(e)}")
    except Exception as e:
        raise Exception(f"Ошибка при работе с GigaChat: {str(e)}")


def main():
    """Главная функция CLI."""
    parser = argparse.ArgumentParser(
        description="Анализ изображений блюд с помощью OpenAI Vision API и генерация рецептов с помощью GigaChat-2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python dish_analyzer.py Блюдо_1.jpg
  python dish_analyzer.py path/to/image.png --output result.json
  
Приложение выполняет три этапа:
  1. Анализ изображения блюда с помощью OpenAI Vision API
  2. Генерация рецепта, этапов приготовления, КБЖУ и промпта для изображения с помощью GigaChat-2
  3. Генерация изображения блюда с помощью OpenAI DALL-E на основе промпта от GigaChat
        """
    )
    
    parser.add_argument(
        "image",
        type=str,
        help="Путь к изображению блюда"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Путь для сохранения результата (по умолчанию выводится в stdout)"
    )
    
    parser.add_argument(
        "--env",
        type=str,
        default=".env",
        help="Путь к файлу .env (по умолчанию: .env)"
    )
    
    parser.add_argument(
        "--json",
        action="store_true",
        help="Вывести результат в формате JSON (по умолчанию - текстовый формат)"
    )
    
    parser.add_argument(
        "--image-output", "-i",
        type=str,
        help="Путь для сохранения сгенерированного изображения (по умолчанию: {имя_исходного_файла}_generated.png)"
    )
    
    args = parser.parse_args()
    
    # Загружаем переменные окружения
    load_dotenv(args.env)
    
    # Получаем API ключ
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Ошибка: OPENAI_API_KEY не найден в переменных окружения.", file=sys.stderr)
        print(f"Убедитесь, что файл '{args.env}' существует и содержит OPENAI_API_KEY=your_key", file=sys.stderr)
        sys.exit(1)
    
    # Проверяем существование файла изображения
    if not os.path.exists(args.image):
        print(f"Ошибка: Файл '{args.image}' не существует.", file=sys.stderr)
        sys.exit(1)
    
    # Получаем настройки OpenAI
    openai_model = os.getenv("OPENAI_MODEL", "gpt-4o")
    openai_max_tokens = int(os.getenv("OPENAI_MAX_TOKENS", "500"))
    openai_temperature = float(os.getenv("OPENAI_TEMPERATURE", "0.3"))
    
    # Анализируем блюдо с помощью OpenAI
    print(f"Анализ изображения блюда с помощью OpenAI Vision ({openai_model})...", file=sys.stderr)
    dish_info = analyze_dish(
        args.image,
        api_key,
        model=openai_model,
        max_tokens=openai_max_tokens,
        temperature=openai_temperature
    )
    
    # Получаем настройки GigaChat
    gigachat_api_key = os.getenv("GIGACHAT_API_KEY")
    gigachat_scope = os.getenv("GIGACHAT_SCOPE", "GIGACHAT_API_PERS")
    gigachat_api_url = os.getenv("GIGACHAT_API_URL", "https://gigachat.devices.sberbank.ru/api/v1/chat/completions")
    gigachat_token_url = os.getenv("GIGACHAT_TOKEN_URL", "https://ngw.devices.sberbank.ru:9443/api/v2/oauth")
    gigachat_model = os.getenv("GIGACHAT_MODEL", "GigaChat-2")
    gigachat_max_tokens = int(os.getenv("GIGACHAT_MAX_TOKENS", "4000"))
    gigachat_temperature = float(os.getenv("GIGACHAT_TEMPERATURE", "0.7"))
    gigachat_repetition_penalty = float(os.getenv("GIGACHAT_REPETITION_PENALTY", "1.1"))
    gigachat_verify_ssl = os.getenv("GIGACHAT_VERIFY_SSL", "false").lower() == "true"
    
    # Генерируем рецепт с помощью GigaChat
    if gigachat_api_key:
        print(f"Генерация рецепта с помощью {gigachat_model}...", file=sys.stderr)
        try:
            recipe_data = generate_recipe_with_gigachat(
                dish_info,
                gigachat_api_key,
                gigachat_scope,
                gigachat_api_url,
                gigachat_token_url,
                model=gigachat_model,
                max_tokens=gigachat_max_tokens,
                temperature=gigachat_temperature,
                repetition_penalty=gigachat_repetition_penalty,
                verify_ssl=gigachat_verify_ssl
            )
            
            # Объединяем результаты
            result = {
                "dish_analysis": dish_info,
                "recipe": recipe_data
            }
        except Exception as e:
            print(f"Предупреждение: Не удалось получить рецепт от GigaChat: {e}", file=sys.stderr)
            print("Используется только результат анализа OpenAI.", file=sys.stderr)
            result = {
                "dish_analysis": dish_info,
                "recipe": None,
                "error": str(e)
            }
    else:
        print("Предупреждение: GIGACHAT_API_KEY не найден. Используется только результат анализа OpenAI.", file=sys.stderr)
        result = {
            "dish_analysis": dish_info,
            "recipe": None
        }
    
    # Генерируем изображение, если есть промпт
    recipe_data = result.get("recipe")
    if recipe_data and recipe_data.get("image_prompt"):
        # Получаем настройки для генерации изображения
        image_model = os.getenv("OPENAI_IMAGE_MODEL", "dall-e-2")
        image_size = os.getenv("OPENAI_IMAGE_SIZE", "1024x1024")
        image_quality = os.getenv("OPENAI_IMAGE_QUALITY", "standard")
        image_n = int(os.getenv("OPENAI_IMAGE_N", "1"))
        image_response_format = os.getenv("OPENAI_IMAGE_RESPONSE_FORMAT", "url")
        
        image_prompt = recipe_data["image_prompt"]
        
        # Улучшаем промпт для более фотореалистичного результата
        # Добавляем указания на фотореалистичность, если их еще нет
        enhanced_prompt = image_prompt
        if "photograph" not in image_prompt.lower() and "photo" not in image_prompt.lower():
            enhanced_prompt = f"Photorealistic food photography, {image_prompt}"
        if "realistic" not in image_prompt.lower():
            enhanced_prompt = f"Realistic {enhanced_prompt}"
        
        print(f"Генерация изображения с помощью {image_model}...", file=sys.stderr)
        try:
            # Генерируем изображение
            image_result = generate_image_with_openai(
                prompt=enhanced_prompt,
                api_key=api_key,
                model=image_model,
                size=image_size,
                quality=image_quality,
                n=image_n,
                response_format=image_response_format
            )
            
            # Определяем путь для сохранения изображения
            if args.image_output:
                output_image_path = Path(args.image_output)
            else:
                input_image_path = Path(args.image)
                output_image_path = input_image_path.parent / f"{input_image_path.stem}_generated.png"
            
            # Сохраняем изображение
            if image_result.get("url"):
                download_image_from_url(image_result["url"], str(output_image_path))
            elif image_result.get("b64_json"):
                save_image_from_base64(image_result["b64_json"], str(output_image_path))
            else:
                raise Exception("Не удалось получить изображение из ответа API")
            
            # Добавляем информацию о сгенерированном изображении в результат
            result["generated_image"] = {
                "file_path": str(output_image_path),
                "model": image_model,
                "size": image_size,
                "quality": image_quality
            }
            
            print(f"Изображение сохранено: {output_image_path}", file=sys.stderr)
            
        except Exception as e:
            print(f"Предупреждение: Не удалось сгенерировать изображение: {e}", file=sys.stderr)
            result["generated_image"] = None
            result["image_generation_error"] = str(e)
    
    # Форматируем вывод
    if args.json:
        # JSON формат
        output_text = json.dumps(result, ensure_ascii=False, indent=2)
    else:
        # Текстовый формат
        output_text = format_output_text(result)
    
    # Выводим или сохраняем результат
    if args.output:
        try:
            # Создаем директорию, если её нет
            output_file = Path(args.output)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(output_text)
            print(f"Результат сохранен в '{args.output}'", file=sys.stderr)
        except Exception as e:
            print(f"Ошибка при сохранении файла: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        print(output_text)


if __name__ == "__main__":
    main()

