# работа с GigaChat API

import requests
import time
import uuid
import urllib3
from app.config import (
    GIGACHAT_API_KEY,
    GIGACHAT_API_URL,
    GIGACHAT_TOKEN_URL,
    GIGACHAT_SCOPE,
    GIGACHAT_MODEL,
    GIGACHAT_VERIFY_SSL
)
from app.history import get_history, add_assistant_message

# Отключаем предупреждения о небезопасных SSL запросах, если проверка отключена
if not GIGACHAT_VERIFY_SSL:
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Кэш для Access token
_access_token = None
_token_expires_at = 0
TOKEN_LIFETIME = 30 * 60  # 30 минут в секундах


def get_access_token() -> str:
    """
    Получает Access token для авторизации запросов к GigaChat API.
    Токен кэшируется и обновляется при необходимости (действует 30 минут).
    
    Returns:
        Access token
    """
    global _access_token, _token_expires_at
    
    # Проверяем наличие необходимых данных
    if not GIGACHAT_API_KEY:
        raise Exception("GIGACHAT_API_KEY не установлен. Проверьте файл .env")
    
    if not GIGACHAT_SCOPE:
        raise Exception("GIGACHAT_SCOPE не установлен. Проверьте файл .env")
    
    # Проверяем, не истек ли токен (оставляем запас в 1 минуту)
    current_time = time.time()
    if _access_token and current_time < (_token_expires_at - 60):
        return _access_token
    
    # Получаем новый токен
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded',
        'Accept': 'application/json',
        'RqUID': str(uuid.uuid4()),
        'Authorization': f'Basic {GIGACHAT_API_KEY}'
    }
    
    payload = {
        'scope': GIGACHAT_SCOPE
    }
    
    try:
        response = requests.post(
            GIGACHAT_TOKEN_URL,
            headers=headers,
            data=payload,
            timeout=30,
            verify=GIGACHAT_VERIFY_SSL
        )
        
        # Детальная обработка ошибок
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
        _access_token = data.get('access_token')
        
        if not _access_token:
            raise ValueError("Access token не найден в ответе API")
        
        # Сохраняем время истечения токена
        _token_expires_at = current_time + TOKEN_LIFETIME
        
        return _access_token
        
    except requests.exceptions.HTTPError as e:
        error_detail = ""
        if hasattr(e.response, 'text'):
            try:
                error_data = e.response.json()
                error_detail = f" Детали: {error_data}"
            except:
                error_detail = f" Ответ сервера: {e.response.text[:200]}"
        raise Exception(f"HTTP ошибка при получении Access token: {str(e)}{error_detail}")
    except requests.exceptions.RequestException as e:
        raise Exception(f"Ошибка при получении Access token: {str(e)}")
    except KeyError as e:
        raise Exception(f"Ошибка при обработке ответа API токена: {str(e)}")
    except Exception as e:
        raise Exception(f"Неожиданная ошибка при получении токена: {str(e)}")


def send_message_to_gigachat() -> str:
    """
    Отправляет историю диалога в GigaChat API и возвращает ответ.
    Использует всю историю диалога для контекста.
    
    Returns:
        Ответ от GigaChat
    """
    # Получаем историю диалога
    messages = get_history()
    
    # Получаем Access token
    try:
        access_token = get_access_token()
    except Exception as e:
        return f"Ошибка авторизации: {str(e)}"
    
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'Authorization': f'Bearer {access_token}'
    }
    
    payload = {
        "model": GIGACHAT_MODEL,
        "messages": messages,
        "n": 1,
        "stream": False,
        "max_tokens": 512,
        "temperature": 0.1,
        "repetition_penalty": 1,
        "update_interval": 0
    }
    
    try:
        response = requests.post(
            GIGACHAT_API_URL,
            headers=headers,
            json=payload,
            timeout=30,
            verify=GIGACHAT_VERIFY_SSL
        )
        response.raise_for_status()
        
        data = response.json()
        # Извлекаем ответ из структуры ответа API
        if "choices" in data and len(data["choices"]) > 0:
            response_text = data["choices"][0]["message"]["content"]
            # Добавляем ответ ассистента в историю
            add_assistant_message(response_text)
            return response_text
        else:
            return "Ошибка: не удалось получить ответ от API"
            
    except requests.exceptions.RequestException as e:
        return f"Ошибка при обращении к API: {str(e)}"
    except KeyError as e:
        return f"Ошибка при обработке ответа API: {str(e)}"
    except Exception as e:
        return f"Неожиданная ошибка: {str(e)}"
