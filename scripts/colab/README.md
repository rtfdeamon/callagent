# XTTS Fine-tune on Andreev Voice — Инструкция

## Требования
- Google Drive с свободным местом ~5GB
- Google Colab (бесплатный T4 GPU или Pro с L4/A100)
- Датасет `data/andreev_dataset/` (уже готов: 635 сегментов, metadata.csv)

## Пошаговая инструкция

### 1. Загрузить датасет в Google Drive
```bash
# На локальной машине (linux):
rclone copy data/andreev_dataset/ drive:andreev_dataset/ -P
# или через веб-интерфейс: drag-drop папку в MyDrive
```

### 2. Открыть notebook в Colab
1. Открыть https://colab.research.google.com
2. File → Upload notebook → `XTTS_Andreev_Finetune.ipynb`
3. Runtime → Change runtime type → T4 GPU (free) или L4/A100 (Pro)

### 3. Запустить все ячейки (Runtime → Run all)
- Ячейка 1-4: setup (~5 мин)
- Ячейка 5: подготовка датасета (~5 мин)
- Ячейка 6: **тренировка (6-12 часов на T4, 2-4 часа на L4)**

### 4. Мониторинг
Colab может отваливаться каждые 12 часов (free) или 24 часа (Pro). 
- Чекпойнты сохраняются каждые 10000 шагов в Drive
- Можно возобновить с последнего checkpoint

### 5. Скачать модель
После тренировки скопировать из Drive:
```
/MyDrive/xtts_andreev_finetune/best_model.pth
/MyDrive/xtts_andreev_finetune/config.json
```

## Интеграция в проект
```bash
# Заменить в проекте:
cp best_model.pth models/xtts_v2/model.pth
cp config.json models/xtts_v2/config.json

# Перегенерировать все фразы:
docker run ... python generate_phrases.py
```

## Ожидаемое качество
- Zero-shot XTTS (текущий): 60-75% сходства
- Fine-tune 10 epochs (3-6h): 80-85% сходства
- Fine-tune 30 epochs (10-15h): 87-92% сходства

## Если T4 OOM
- `batch_size`: 3 → 2 → 1
- `grad_accum_steps`: 8 → 16 (чтобы эффективный batch был тот же)
- `max_wav_length`: 255995 → 220500 (10s)
