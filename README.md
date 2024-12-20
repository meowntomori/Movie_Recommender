# 🎬 Movie Recommender
**Movie Recommender** - это проект, который поможет вам найти идеальный фильм, основываясь на ваших предпочтениях и истории просмотров!

## 📂 Структура проекта

- [**`main`**](https://github.com/meowntomori/movie_recommender) - Основная ветка, где хранится рабочая версия проекта.   
- [**`data`**](https://github.com/meowntomori/movie_recommender/tree/master/data) - данные
- [**`models`**](https://github.com/meowntomori/movie_recommender/tree/master/models) - модели
- [**`templates`**](https://github.com/meowntomori/movie_recommender/tree/master/templates) - шаблоны
- [**`static`**](https://github.com/meowntomori/movie_recommender/tree/master/static) - статические файлы

## Установка и запуск

**Склонируйте репозиторий:**

```
git clone https://github.com/meowntomori/movie_recommender.git
```

У вас появятся несколько папок:

**data** - папка с данными, которые используются для обучения моделей
**models** - папка с моделями, которые обучаются на основе данных
**recommender** - папка с рекомендательной системой
**templates** - папка с шаблонами HTML для веб-приложения
**static** - папка с статическими файлами (CSS, JavaScript)

Для начала работы необходимо создать виртуальное окружение:

```
python -m venv venv
```

В каждой папке лежит файл **requirements.txt** со всеми зависимостями. Необходимо пройтись по каждой папке и выполнять команду для установки всех необходимых библиотек:

```
pip install -r requirements.txt
```

Запуск скриптов:

Для запуска рекомендательной системы необходимо запустить скрипт app.py

## 🛠️ Технологии проекта
**Язык программирования:** Python
**Библиотеки и фреймворки:** Flask, Pandas, NumPy, scikit-learn, nltk, pymorphy2
**Frontend:** HTML, CSS, JavaScript

## 📈 Перспективы развития
Расширение функционала модели для рекомендаций фильмов на основе дополнительных параметров.
Добавление возможности предсказания рейтингов фильмов на основе отзывов пользователей.
Увеличение набора данных для обучения моделей, включая данные из других источников.
Улучшение дизайна сайта и интерфейса для повышения удобства использования.
