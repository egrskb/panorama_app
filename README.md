# PANORAMA

Минималистичный рефакторинг исходного одnofайлoвого приложения
`sweep_analyzer.py` в структуру с пакетами и конфигами.

## Установка

```bash
pip install -e .
```

## Запуск

```bash
# через установленный консольный скрипт
panorama
# или напрямую модуль
python -m panorama
```

Приложение загружает настройки из `configs/app.default.json` и
пользовательского файла `user.json`:

* Linux: `~/.config/PANORAMA/user.json`
* Windows: `%APPDATA%/PANORAMA/user.json`

## libhackrf_qsa.so

Для режима "library" требуется собрать и положить рядом с исполняемым
файлом библиотеку `libhackrf_qsa.so`. Сборка остаётся такой же, как в
оригинальном проекте.
