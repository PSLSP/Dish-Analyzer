#!/usr/bin/env python3
"""
Flask веб-приложение для анализа изображений блюд.
"""

import os
import json
import base64
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_from_directory, Response, stream_with_context
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

# Импортируем функции из dish_analyzer
from dish_analyzer import (
    analyze_dish,
    generate_recipe_with_gigachat,
    generate_image_with_openai,
    download_image_from_url,
    save_image_from_base64,
    get_gigachat_access_token
)

# Загружаем переменные окружения
load_dotenv()

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['GENERATED_FOLDER'] = 'static/generated'

# Создаем необходимые директории
Path(app.config['UPLOAD_FOLDER']).mkdir(exist_ok=True)
Path(app.config['GENERATED_FOLDER']).mkdir(parents=True, exist_ok=True)

# Получаем настройки из .env
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GIGACHAT_API_KEY = os.getenv("GIGACHAT_API_KEY")
GIGACHAT_SCOPE = os.getenv("GIGACHAT_SCOPE", "GIGACHAT_API_PERS")
GIGACHAT_API_URL = os.getenv("GIGACHAT_API_URL", "https://gigachat.devices.sberbank.ru/api/v1/chat/completions")
GIGACHAT_TOKEN_URL = os.getenv("GIGACHAT_TOKEN_URL", "https://ngw.devices.sberbank.ru:9443/api/v2/oauth")
GIGACHAT_MODEL = os.getenv("GIGACHAT_MODEL", "GigaChat-2")
GIGACHAT_MAX_TOKENS = int(os.getenv("GIGACHAT_MAX_TOKENS", "4000"))
GIGACHAT_TEMPERATURE = float(os.getenv("GIGACHAT_TEMPERATURE", "0.7"))
GIGACHAT_REPETITION_PENALTY = float(os.getenv("GIGACHAT_REPETITION_PENALTY", "1.1"))
GIGACHAT_VERIFY_SSL = os.getenv("GIGACHAT_VERIFY_SSL", "false").lower() == "true"

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
OPENAI_MAX_TOKENS = int(os.getenv("OPENAI_MAX_TOKENS", "500"))
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.3"))

OPENAI_IMAGE_MODEL = os.getenv("OPENAI_IMAGE_MODEL", "dall-e-2")
OPENAI_IMAGE_SIZE = os.getenv("OPENAI_IMAGE_SIZE", "1024x1024")
OPENAI_IMAGE_QUALITY = os.getenv("OPENAI_IMAGE_QUALITY", "standard")
OPENAI_IMAGE_N = int(os.getenv("OPENAI_IMAGE_N", "1"))
OPENAI_IMAGE_RESPONSE_FORMAT = os.getenv("OPENAI_IMAGE_RESPONSE_FORMAT", "url")

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}


def allowed_file(filename):
    """Проверяет, разрешен ли формат файла."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    """Главная страница."""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    """Загрузка файла изображения."""
    if 'file' not in request.files:
        return jsonify({'error': 'Файл не найден'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Файл не выбран'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Конвертируем изображение в base64 для отображения
        with open(filepath, 'rb') as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')
            image_ext = Path(filepath).suffix.lower().replace('.', '')
            image_base64 = f"data:image/{image_ext};base64,{image_data}"
        
        return jsonify({
            'success': True,
            'filename': filename,
            'filepath': filepath,
            'image_base64': image_base64
        })
    
    return jsonify({'error': 'Неподдерживаемый формат файла'}), 400


def send_event(event_type, data):
    """Отправляет SSE событие."""
    return f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.route('/analyze', methods=['POST'])
def analyze():
    """Анализ блюда и генерация рецепта с отправкой событий о прогрессе."""
    data = request.json
    filepath = data.get('filepath')
    
    if not filepath or not os.path.exists(filepath):
        return jsonify({'error': 'Файл не найден'}), 400
    
    def generate():
        try:
            # Этап 1: Анализ изображения с помощью OpenAI
            yield send_event('progress', {'message': 'Анализ фотографии блюда', 'stage': 1})
            
            dish_info = analyze_dish(
                filepath,
                OPENAI_API_KEY,
                model=OPENAI_MODEL,
                max_tokens=OPENAI_MAX_TOKENS,
                temperature=OPENAI_TEMPERATURE
            )
            
            result = {
                'dish_analysis': dish_info,
                'recipe': None,
                'generated_image': None
            }
            
            # Этап 2: Генерация рецепта с помощью GigaChat
            if GIGACHAT_API_KEY:
                try:
                    yield send_event('progress', {'message': 'Создание рецепта и таблицы КБЖУ/ХЕ', 'stage': 2})
                    
                    recipe_data = generate_recipe_with_gigachat(
                        dish_info,
                        GIGACHAT_API_KEY,
                        GIGACHAT_SCOPE,
                        GIGACHAT_API_URL,
                        GIGACHAT_TOKEN_URL,
                        model=GIGACHAT_MODEL,
                        max_tokens=GIGACHAT_MAX_TOKENS,
                        temperature=GIGACHAT_TEMPERATURE,
                        repetition_penalty=GIGACHAT_REPETITION_PENALTY,
                        verify_ssl=GIGACHAT_VERIFY_SSL
                    )
                    result['recipe'] = recipe_data
                    
                    # Этап 3: Генерация изображения
                    if recipe_data and recipe_data.get("image_prompt"):
                        yield send_event('progress', {'message': 'Генерация изображения "как должно выглядеть блюдо"', 'stage': 3})
                        
                        image_prompt = recipe_data["image_prompt"]
                        
                        # Улучшаем промпт для фотореалистичности
                        enhanced_prompt = image_prompt
                        if "photograph" not in image_prompt.lower() and "photo" not in image_prompt.lower():
                            enhanced_prompt = f"Photorealistic food photography, {image_prompt}"
                        if "realistic" not in image_prompt.lower():
                            enhanced_prompt = f"Realistic {enhanced_prompt}"
                        
                        try:
                            image_result = generate_image_with_openai(
                                prompt=enhanced_prompt,
                                api_key=OPENAI_API_KEY,
                                model=OPENAI_IMAGE_MODEL,
                                size=OPENAI_IMAGE_SIZE,
                                quality=OPENAI_IMAGE_QUALITY,
                                n=OPENAI_IMAGE_N,
                                response_format=OPENAI_IMAGE_RESPONSE_FORMAT
                            )
                            
                            # Сохраняем изображение
                            image_filename = f"generated_{Path(filepath).stem}.png"
                            image_path = os.path.join(app.config['GENERATED_FOLDER'], image_filename)
                            
                            if image_result.get("url"):
                                download_image_from_url(image_result["url"], image_path)
                            elif image_result.get("b64_json"):
                                save_image_from_base64(image_result["b64_json"], image_path)
                            
                            # Конвертируем в base64 для отображения
                            with open(image_path, 'rb') as f:
                                img_data = base64.b64encode(f.read()).decode('utf-8')
                                generated_image_base64 = f"data:image/png;base64,{img_data}"
                            
                            result['generated_image'] = {
                                'image_base64': generated_image_base64,
                                'filename': image_filename
                            }
                        except Exception as e:
                            result['image_error'] = str(e)
                            
                except Exception as e:
                    result['recipe_error'] = str(e)
            
            # Отправляем финальный результат
            yield send_event('complete', result)
            
        except Exception as e:
            yield send_event('error', {'error': str(e)})
    
    return Response(stream_with_context(generate()), mimetype='text/event-stream')


@app.route('/static/generated/<filename>')
def generated_image(filename):
    """Отдача сгенерированных изображений."""
    return send_from_directory(app.config['GENERATED_FOLDER'], filename)


if __name__ == '__main__':
    if not OPENAI_API_KEY:
        print("Ошибка: OPENAI_API_KEY не найден в переменных окружения.")
        print("Убедитесь, что файл .env существует и содержит OPENAI_API_KEY=your_key")
        exit(1)
    
    app.run(debug=True, host='0.0.0.0', port=5000)

