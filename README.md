# 🍽️ Dish Analyzer / Анализатор блюд

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**CLI и веб-приложение** для анализа фотографий блюд: распознавание блюда и ингредиентов (OpenAI Vision), генерация пошагового рецепта с КБЖУ и хлебными единицами (GigaChat-2), опциональная генерация изображения блюда (OpenAI DALL-E).

---

## ✨ Возможности

- **Анализ по фото** — загрузите изображение блюда и получите название, тип кухни и список ингредиентов
- **Рецепт и КБЖУ** — пошаговый рецепт, таблица калорий/белков/жиров/углеводов и хлебных единиц (на порцию и на 100 г)
- **Генерация изображения** — фотореалистичная визуализация «как должно выглядеть блюдо» (DALL-E)
- **Два интерфейса** — командная строка для скриптов и автоматизации, веб-интерфейс для интерактивного использования
- **Гибкая настройка** — все параметры моделей и API задаются через `.env`

---

## 🛠 Стек

| Этап | Сервис | Назначение |
|------|--------|------------|
| 1 | **OpenAI Vision** (gpt-4o) | Анализ изображения → название, ингредиенты, кухня |
| 2 | **GigaChat-2** | Рецепт, этапы приготовления, КБЖУ/ХЕ, промпт для изображения |
| 3 | **OpenAI DALL-E** | Генерация изображения по промпту от GigaChat |

---

## 📋 Требования

- **Python 3.9+**
- API-ключи:
  - [OpenAI API](https://platform.openai.com/api-keys) (обязательно) — для анализа изображения и генерации картинки
  - [GigaChat](https://developers.sber.ru/portal/products/gigachat-api) (опционально) — для рецепта; без него возвращается только анализ блюда

---

## 🚀 Установка

### 1. Клонирование репозитория

```bash
git clone https://github.com/YOUR_USERNAME/dish-analyzer.git
cd dish-analyzer
```

### 2. Виртуальное окружение (рекомендуется)

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate
```

### 3. Установка зависимостей

```bash
pip install -r requirements.txt
```

### 4. Настройка переменных окружения

Создайте файл `.env` в корне проекта (скопируйте из примера и заполните ключи):

```env
# Обязательно
OPENAI_API_KEY=sk-your-openai-key-here

# Опционально (для рецепта и КБЖУ)
GIGACHAT_API_KEY=your_gigachat_api_key_here
GIGACHAT_SCOPE=GIGACHAT_API_PERS
```

Файл `.env` не должен попадать в репозиторий (добавлен в `.gitignore`).

---

## ⚙️ Конфигурация (.env)

### OpenAI — анализ изображения (по умолчанию)

| Переменная | По умолчанию | Описание |
|------------|--------------|----------|
| `OPENAI_MODEL` | `gpt-4o` | Модель для анализа фото |
| `OPENAI_MAX_TOKENS` | `500` | Лимит токенов ответа |
| `OPENAI_TEMPERATURE` | `0.3` | Температура (низкая для стабильного JSON) |

### OpenAI — генерация изображения

| Переменная | По умолчанию | Описание |
|------------|--------------|----------|
| `OPENAI_IMAGE_MODEL` | `dall-e-2` | Модель: `dall-e-2` или `dall-e-3` |
| `OPENAI_IMAGE_SIZE` | `1024x1024` | Размер (для dall-e-3: `1024x1024`, `1792x1024`, `1024x1792`) |
| `OPENAI_IMAGE_QUALITY` | `standard` | Только для dall-e-3: `standard` или `hd` |
| `OPENAI_IMAGE_N` | `1` | Количество изображений (для dall-e-3 всегда 1) |
| `OPENAI_IMAGE_RESPONSE_FORMAT` | `url` | `url` или `b64_json` |

### GigaChat

| Переменная | По умолчанию | Описание |
|------------|--------------|----------|
| `GIGACHAT_API_URL` | (Sber) | URL API чата |
| `GIGACHAT_TOKEN_URL` | (Sber) | URL для OAuth-токена |
| `GIGACHAT_MODEL` | `GigaChat-2` | Модель |
| `GIGACHAT_MAX_TOKENS` | `4000` | Лимит токенов |
| `GIGACHAT_TEMPERATURE` | `0.7` | Температура |
| `GIGACHAT_REPETITION_PENALTY` | `1.1` | Штраф за повторения |
| `GIGACHAT_VERIFY_SSL` | `false` | Проверка SSL (для корпоративного API часто `false`) |

---

## 📖 Использование

### CLI

**Базовый запуск (результат в консоль):**

```bash
python dish_analyzer.py путь/к/изображению.jpg
```

**Сохранение в файл:**

```bash
python dish_analyzer.py photo.jpg --output result.txt
```

**Вывод в JSON:**

```bash
python dish_analyzer.py photo.jpg --json
python dish_analyzer.py photo.jpg --json --output result.json
```

**Сохранение сгенерированного изображения:**

```bash
python dish_analyzer.py photo.jpg --image-output my_dish.png
python dish_analyzer.py photo.jpg -i output/image.png
```

**Другой файл окружения:**

```bash
python dish_analyzer.py photo.jpg --env .env.production
```

**Справка по аргументам:**

```bash
python dish_analyzer.py --help
```

### Веб-интерфейс

```bash
python app.py
```

Откройте в браузере: **http://localhost:5000**

- Загрузка изображения (перетаскивание или выбор файла)
- Автоматический или ручной запуск анализа
- Отображение: название, ингредиенты, кухня, рецепт, этапы, КБЖУ/ХЕ, сгенерированное изображение
- Кнопка «Очистить» для сброса

---

## 📂 Формат вывода

### Текстовый вывод (по умолчанию)

- Блок «Анализ блюда»: название, кухня, уверенность, ингредиенты  
- Рецепт: метаданные, ингредиенты с количествами, этапы приготовления  
- Таблица КБЖУ и хлебных единиц (на порцию и на 100 г)  
- Промпт для генерации изображения и путь к сгенерированному файлу (если есть)

### JSON (`--json`)

Структура:

```json
{
  "dish_analysis": {
    "dish_name": "Паста карбонара",
    "ingredients": ["спагетти", "бекон", "яйца", "пармезан"],
    "cuisine": "итальянская",
    "confidence": 0.82
  },
  "recipe": {
    "recipe": { "name", "cuisine", "servings", "prep_time", "cook_time", "total_time", "ingredients": [...] },
    "cooking_steps": [ { "step_number", "description", "duration" } ],
    "nutrition": {
      "per_serving": { "calories", "protein", "fat", "carbs", "bread_units" },
      "per_100g": { ... }
    },
    "image_prompt": "Professional food photography of ..."
  },
  "generated_image": { "file_path", "model", "size", "quality" }
}
```

Если `GIGACHAT_API_KEY` не задан, в ответе будет только `dish_analysis`.

### Поддерживаемые форматы изображений

JPEG (`.jpg`, `.jpeg`), PNG (`.png`), GIF (`.gif`), WebP (`.webp`). Максимальный размер загружаемого файла в веб-интерфейсе — 16 МБ.

---

## 📁 Структура проекта

```
dish-analyzer/
├── app.py              # Flask веб-приложение
├── dish_analyzer.py    # Ядро: CLI, Vision, GigaChat, DALL-E
├── requirements.txt
├── .env                # Ваши ключи (не коммитить!)
├── .env.example        # Пример переменных (опционально)
├── README.md
├── templates/
│   └── index.html      # Страница веб-интерфейса
├── uploads/            # Загруженные файлы (веб)
└── static/generated/   # Сгенерированные изображения (веб)
```

---

## ⚠️ Важно

- **КБЖУ и хлебные единицы** — оценочные, рассчитанные моделью. Не заменяют консультацию врача или диетолога. При диабете и строгих диетах используйте только после консультации специалиста.
- **API-ключи** храните только в `.env` и не публикуйте их в репозитории.
- Для **GigaChat** может потребоваться отключение проверки SSL (`GIGACHAT_VERIFY_SSL=false`) в зависимости от окружения.

---

## 📄 Лицензия

MIT (или укажите свою лицензию).

---

## 🤝 Участие в разработке

Pull requests приветствуются. Для крупных изменений лучше сначала обсудить их в Issue.
