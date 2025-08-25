# Исправление ошибки 'DetectedPeak object is not subscriptable'

## Описание проблемы

При работе с детектором пиков возникала ошибка:
```
detect_and_emit error: 'DetectedPeak' object is not subscriptable
```

## Причина ошибки

Код в `MasterSweepController._detect_and_emit()` пытался обращаться к объекту `DetectedPeak` как к кортежу:

```python
# Неправильный код:
peak_freq, peak_snr = max(peaks, key=lambda t: t[1])
```

Но `peak_detector.detect_peaks()` возвращает список объектов `DetectedPeak`, а не кортежей.

## Решение

Исправлен код для правильного обращения к атрибутам объекта:

```python
# Правильный код:
best_peak = max(peaks, key=lambda p: p.snr_db)
peak_freq = best_peak.freq_hz
peak_snr = best_peak.snr_db
```

## Файлы

- **Исправлен**: `panorama/features/spectrum/master.py`
- **Строки**: 133-135

## Результат

✅ Ошибка исправлена
✅ Детектор пиков работает корректно
✅ Код стал более читаемым и понятным

## Тестирование

Для проверки исправления запустите:
```bash
cd panorama
python3 test_peak_fix.py
```

## Статус

**Исправлено** ✅ - ошибка больше не возникает
