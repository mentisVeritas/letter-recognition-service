# Letter Recognition Service

Сервис распознавания латинских букв A–Z: рисуете букву на canvas → модель предсказывает её.

## Запуск

### Локально (без Docker)

1. Установите зависимости:
   ```bash
   pip install -r requirements-docker.txt
   ```
   Для обучения модели используйте полный `requirements.txt`.

2. Убедитесь, что обученная модель лежит в `model/cnn.pth` (см. раздел «Обучение»).

3. Запустите сервер:
   ```bash
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

4. Откройте в браузере: http://127.0.0.1:8000

### Docker

```bash
docker build -t letter-recognition .
docker run -p 8000:8000 letter-recognition
```

Откройте http://127.0.0.1:8000

**Важно:** в образ нужно положить файл `model/cnn.pth` (обученная модель). Если его нет, при первом запросе к `/v1/predict` сервер вернёт 503 с подсказкой.

## API

- **GET /** — отдаёт HTML-страницу с canvas для рисования.
- **POST /v1/predict** — предсказание буквы по изображению.

Тело запроса (JSON):
```json
{ "image": "<base64-строка изображения (data URL или сырой base64)>" }
```

Ответ:
```json
{
  "prediction": "A",
  "top5": [
    { "letter": "A", "prob": 0.92 },
    { "letter": "H", "prob": 0.03 },
    ...
  ]
}
```

Изображение должно быть в оттенках серого, после препроцессинга оно приводится к 28×28, инвертируется (белый фон → чёрный в данных).

## Обучение модели

1. Положите данные в `data/`:
   - `letters_train_fixed.csv` — обучающая выборка (первый столбец — метка буквы 1–26, остальные 784 — пиксели 28×28).

2. Откройте ноутбук и выполните все ячейки:
   ```bash
   jupyter notebook notebooks/train.ipynb
   ```

3. Сохранённый файл `model/cnn.pth` скопируйте в корень проекта (или оставьте там, где ноутбук его сохраняет). Сервис ожидает путь `model/cnn.pth` относительно каталога `app/`.

## Структура проекта

```
letter-recognition-service/
├── app/
│   ├── main.py      # FastAPI: роуты, CORS, обработка ошибок
│   ├── model.py     # CNN и загрузка весов (cnn.pth)
│   ├── schemas.py   # Pydantic-модели запроса/ответа
│   └── utils.py     # препроцессинг base64 → 28×28
├── frontend/
│   └── index.html   # Canvas + запрос к /v1/predict
├── model/
│   └── cnn.pth      # веса модели (создаётся при обучении)
├── data/            # CSV для обучения
├── notebooks/
│   └── train.ipynb  # обучение CNN
├── requirements.txt
├── requirements-docker.txt
└── Dockerfile
```
