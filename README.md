# Letter·AI — Сервис распознавания рукописных букв

CNN-модель для классификации рукописных букв A–Z, упакованная в FastAPI + Docker с веб-интерфейсом.

---

## Структура проекта

```
letter-recognition-service/
├── app/
│   ├── main.py        # FastAPI — роуты, инференс, статика
│   ├── model.py       # Загрузка модели
│   ├── schemas.py     # Pydantic схемы
│   └── utils.py       # Утилиты предобработки
├── frontend/
│   └── index.html     # Веб-интерфейс с canvas-доской
├── model/
│   └── letter_cnn.pth # Сериализованная CNN модель
├── notebooks/
│   └── train.ipynb    # Обучение модели
├── requirements.txt        # Локальные зависимости
├── requirements-docker.txt # CPU-only torch для Docker
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

## Модель

| Параметр | Значение |
|----------|----------|
| Архитектура | CNN (3 conv блока + 2 FC слоя) |
| Датасет | Letters dataset, 26 классов (A–Z) |
| Accuracy | 94.3% |
| ROC-AUC | 99.85% (macro OvR) |
| Фреймворк | PyTorch |

### Архитектура

```
Input: 1×28×28
  Conv2d(1→32) + ReLU + MaxPool  →  32×14×14
  Conv2d(32→64) + ReLU + MaxPool →  64×7×7
  Conv2d(64→128) + ReLU          → 128×7×7
  Flatten
  Linear(6272→256) + ReLU + Dropout(0.5)
  Linear(256→26)
Output: 26 классов
```

---

## API

### GET /health
```json
{
  "status": "ok",
  "device": "cpu",
  "classes": ["A", "B", ..., "Z"],
  "model_path": "model/letter_cnn.pth"
}
```

### POST /predict
Принимает 784 пикселя (0–255):
```json
{ "pixels": [0.0, 255.0, ...] }
```

### POST /predict/image
Принимает PNG/JPG файл через multipart/form-data.

**Ответ обоих эндпоинтов:**
```json
{
  "predicted_letter": "A",
  "predicted_index": 0,
  "confidence": 0.9821,
  "probabilities": { "A": 0.9821, "B": 0.003, ... }
}
```

---

## Запуск локально

```bash
# Установить зависимости
pip install -r requirements.txt

# Запустить сервер
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Открыть: http://localhost:8000

---

## Запуск через Docker

```bash
docker-compose up --build
```

Открыть: http://localhost:8000

---

## Деплой на Digital Ocean

```bash
# 1. На дроплете установить Docker
apt-get update -y && apt-get install -y docker.io docker-compose

# 2. Клонировать репозиторий
git clone https://github.com/<username>/letter-recognition-service.git /opt/app
cd /opt/app

# 3. Запустить
docker-compose up -d --build
ufw allow 8000/tcp
```

Сервис доступен на: http://<IP>:8000

---

## Тест через curl

```bash
# Health check
curl http://localhost:8000/health

# Через изображение
curl -X POST http://localhost:8000/predict/image \
  -F "file=@my_letter.png"

# Через массив пикселей
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"pixels": [0,0,...,255]}'
```
