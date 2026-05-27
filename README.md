# second_llm_rag
Мой второй, более масштабный чем первый, проект по ML, где я сделал LLM-сервис с RAG-пайплайном.

Проект реализуется на основе моего [первого проекта по LLM](https://github.com/Alisa-gh2/first_llm_wth_rag) . 

В файле [ANNOTATION_FOR_PROJECT.md](https://github.com/Alisa-gh2/second_llm_rag/blob/main/ANNOTATION_FOR_PROJECT.md) находится подробное описание проекта и того, с чего и как мы начинали.

Сами файлы хранятся в [docs/](https://github.com/Alisa-gh2/second_llm_rag/blob/main/docs).

Проект работает с тремя форматами данных: `.md`, `.pdf`, `.txt`. Перед чанкингом все файлы из корпуса текстов предобрабатываются: они приводятся к общему формату и чистятся от лишнего, например от лишних табуляций и пробелов и разрывов слов из-за переносов. После этого файлы сохраняются в [clean_docs/](https://github.com/Alisa-gh2/second_llm_rag/blob/main/clean_docs). 

В файле [DATA_INFO.md](https://github.com/Alisa-gh2/second_llm_rag/blob/main/DATA_INFO.md) более подробно описано как мы работаем и с какими данными. 


# Установка

## Windows 

1. Установите Docker Desktop.
2. Склонируйте репозиторий.
3. Положите очищенные тексты в папку `clean_docs/` (формат .txt).
4. Создайте файл `.env` с содержимым: OPENROUTER_API_KEY=YOUR_API-KEY

5. Запустите индексацию (один раз):
```bash
docker run --rm -v ${PWD}:/app python:3.10-slim bash -c "cd /app && pip install -r requirements.txt && python scripts/index_data.py"
```
(для Windows PowerShell используйте ${PWD})
6. Запустите сервис:

```bash
docker-compose up --build
```
### Использование

POST /ask – вопрос, возвращает ответ и источники.

GET /health – проверка состояния.

Пример curl:
```bash
curl -X POST "http://localhost:8000/ask" -H "Content-Type: application/json" -d '{"question": "Какие правила обработки персональных данных?"}'
```
