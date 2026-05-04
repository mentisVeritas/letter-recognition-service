# Letter·AI — Сервис распознавания рукописных букв

CNN-модель для классификации рукописных букв A–Z, упакованная в FastAPI + Docker с веб-интерфейсом.

---

## Демо

Продовая версия доступна здесь:  
https://letter-recognition-service.onrender.com/

---

## Структура проекта

letter-recognition-service/ ├── app/ │   ├── main.py        # FastAPI — роуты, инференс, статика │   ├── model.py       # Загрузка модели │   ├── schemas.py     # Pydantic схемы │   └── utils.py       # Утилиты предобработки ├── frontend/ │   └── index.html     # Веб-интерфейс с canvas-доской ├── model/ │   └── letter_cnn.pth # Сериализованная CNN модель ├── notebooks/ │   └── train.ipynb    # Обучение модели ├── requirements.txt        # Локальные зависимости ├── requirements-docker.txt # CPU-only torch для Docker ├── Dockerfile ├── docker-compose.yml └── README.md

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

Input: 1×28×28   Conv2d(1→32) + ReLU + MaxPool  →  32×14×14   Conv2d(32→64) + ReLU + MaxPool →  64×7×7   Conv2d(64→128) + ReLU          → 128×7×7   Flatten   Linear(6272→256) + ReLU + Dropout(0.5)   Linear(256→26) Output: 26 классов

---

## API

### GET /health

https://letter-recognition-service.onrender.com/health

Ответ:
json {   "status": "ok",   "device": "cpu",   "classes": ["A", "B", "...", "Z"],   "model_path": "model/letter_cnn.pth" } 

---

### POST /predict

https://letter-recognition-service.onrender.com/predict

Тело запроса:
json { "pixels": [0.0, 255.0, ...] } 

---

### POST /predict/image

https://letter-recognition-service.onrender.com/predict/image

Принимает PNG/JPG через multipart/form-data

---

### Ответ

json {   "predicted_letter": "A",   "predicted_index": 0,   "confidence": 0.9821,   "probabilities": { "A": 0.9821, "B": 0.003 } } 

---

## Запуск локально (DEV)

bash pip install -r requirements.txt uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 

Открыть:
http://localhost:8000

---

## Запуск через Docker (DEV)

bash docker-compose up --build 

Локально:
http://localhost:8000

Прод:
https://letter-recognition-service.onrender.com/

---

## Деплой

Сервис задеплоен на Render:

https://letter-recognition-service.onrender.com/

Автодеплой происходит из GitHub при push в main.

---

## Тест через curl

bash # Health curl https://letter-recognition-service.onrender.com/health  # Изображение curl -X POST https://letter-recognition-service.onrender.com/predict/image \   -F "file=@my_letter.png"  # Пиксели curl -X POST https://letter-recognition-service.onrender.com/predict \   -H "Content-Type: application/json" \   -d '{"pixels": [0,0,...,255]}' 

---

## Важно

- Убедись, что в коде нет захардкоженного localhost
- Render использует переменную $PORT

Пример запуска:
bash uvicorn app.main:app --host 0.0.0.0 --port $PORT 

- Если фронт отдельно — настрой CORS

---

## TODO

- [ ] Улучшить точность модели
- [ ] Добавить поддержку строчных букв
- [ ] Добавить batch inference
- [ ] Задеплоить фронт отдельно (например, Vercel)

---