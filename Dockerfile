# образ с python 3.10
FROM python:3.10-slim

# рабочая директория внутри контейнера
WORKDIR /app

# копируем файл с зависимостями и устанавливаем их
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# копируем весь код проекта
COPY . .

# открываем порт для api
EXPOSE 8000

# команда запуска сервера uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
