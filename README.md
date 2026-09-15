# Titanic: Machine Learning from Disaster

Решение классификационного соревнования Kaggle Titanic. Итоговый лучший результат:
**hard voting ансамбль из 5 моделей (логрег, Random Forest, XGBoost, LightGBM, CatBoost), CV accuracy 0.847**.

## Структура репозитория

Исследование разбито на отдельные ноутбуки по темам, общий код (препроцессинг, CV-харнесс, модели)
вынесен в модули, чтобы не дублироваться между ноутбуками и переиспользоваться в `main.py`.

**Ноутбуки (по порядку исследования):**
- `eda_and_hypotheses.ipynb` - разведочный анализ, статистики, распределения, гипотезы по фичам
- `simple_baseline_and_experimentations.ipynb` - baseline, сравнение простой/полной предобработки, 1-fold vs 5-fold CV
- `classic_models_exploration.ipynb` - классические модели и бустинги с подбором гиперпараметров (логрег, KNN, дерево, Random Forest, XGBoost, LightGBM, CatBoost), feature engineering раунд 2
- `DNN_exploration.ipynb` - PyTorch MLP: базовая архитектура, глубина, BatchNorm, Dropout, оптимизаторы, scheduler, Embedding-слой для категориальных фичей
- `ensembles.ipynb` - усреднение, voting (soft/hard), stacking (логрег/Ridge), попытка добавить DNN в ансамбль
- `optuna_exploration.ipynb` - сравнение ручного grid search с Optuna на XGBoost и CatBoost (вне чеклиста, для себя)

**Модули (общий код, использует и main.py, и ноутбуки):**
- `preprocessing.py` - генерация фичей (Title/TicketGroupSize/HasCabin), заполнение пропусков, `build_features` - полный пайплайн (one-hot + масштабирование) для обучения
- `validation.py` - StratifiedKFold CV (`cross_validate_model`, `evaluate_model`), перебор гиперпараметров (`tune_hyperparameters`), финальное обучение и сабмит (`train_kfold_and_predict`)
- `dnn.py` - `EmbeddingMLP` (PyTorch) и `EmbeddingMLPClassifier` - sklearn-совместимая обёртка, чтобы DNN можно было использовать в `VotingClassifier`/`StackingClassifier`
- `models.py` - фабрика моделей и ансамблей: строит любую модель или ансамбль из `config.yaml` по имени

**Пайплайн для воспроизведения:**
- `config.yaml` - гиперпараметры всех моделей и описание ансамблей (какие модели входят в voting/stacking), выбор модели для запуска
- `main.py` - загружает данные, строит фичи, обучает модель из конфига, считает CV-скор и сохраняет сабмит
- `run_notebooks.py` - прогоняет все ноутбуки проекта по очереди (jupyter nbconvert --execute), чтобы проверить, что всё воспроизводится с нуля

## Установка

```
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

Проект использует CPU-версию PyTorch. Если `pip install` подтянет версию с CUDA (тяжелее, не нужна на CPU),
поставьте torch отдельно с CPU-индекса:
```
pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
```

Данные соревнования не лежат в репозитории (см. `.gitignore`), скачать через Kaggle CLI
(текущая версия kaggle CLI не умеет распаковывать сама, только скачивает zip):
```
kaggle competitions download -c titanic -p data
unzip data/titanic.zip -d data
```
(или вручную с [страницы соревнования](https://www.kaggle.com/competitions/titanic/data) в `data/train.csv`, `data/test.csv`)

## Запуск

```
python main.py
```

Прогонит модель из `config.yaml` (по умолчанию `voting_hard` - лучший результат проекта) через
5-fold Stratified CV, напечатает accuracy и сохранит `submission.csv`.

Чтобы запустить другую модель или ансамбль, поменяйте поле `model` в `config.yaml` на один из ключей
из секций `models` (логрег, knn, decision_tree, random_forest, xgboost, lightgbm, catboost, dnn_embedding)
или `ensembles` (voting_hard, voting_soft, stacking_logreg, stacking_ridge). Гиперпараметры менять там же -
main.py их не подбирает, только использует уже найденные в ноутбуках значения.

Свой файл конфига: `python main.py --config my_config.yaml`.

Чтобы прогнать по CV все модели и ансамбли разом и увидеть сравнительную таблицу (без сабмита),
поставьте `model: all` в конфиге и снова запустите `python main.py`.

Чтобы воспроизвести все эксперименты с нуля (перевыполнить все ноутбуки проекта, а не только
финальный пайплайн):  

```
python run_notebooks.py
```  

Занимает порядка 15 минут, печатает `OK`/`FAILED` по каждому ноутбуку.

## Результаты

CV accuracy (StratifiedKFold, 5 фолдов, random_state=42), по убыванию:

| Модель | CV accuracy | std |
|---|---|---|
| **voting_hard** | **0.8473** | 0.0099 |
| averaging_soft_voting_with_dnn | 0.8462 | - |
| xgboost | 0.8462 | 0.0219 |
| averaging_soft_voting | 0.8451 | 0.0221 |
| lightgbm | 0.8451 | 0.0185 |
| catboost | 0.8440 | 0.0058 |
| voting_hard_with_dnn | 0.8440 | - |
| stacking_ridge | 0.8429 | - |
| dnn_embedding | 0.8406 | 0.0146 |
| stacking_logreg | 0.8406 | - |
| random_forest | 0.8395 | 0.0129 |
| logreg | 0.8339 | 0.0129 |
| decision_tree | 0.8271 | 0.0230 |
| knn | 0.8260 | 0.0222 |

Baseline (простой логрег без feature engineering и нормирования) - 0.799. Полный цикл предобработки,
feature engineering и подбора гиперпараметров поднял его до 0.834, а ансамблирование бустингов и
классических моделей - до 0.847.

**Выводы по проекту:**
- бустинги (XGBoost/LightGBM/CatBoost) уверенно лучше классики (логрег/KNN/дерево) поодиночке
- простое hard voting на 5 разных моделях обошло и лучший одиночный бустинг, и оба варианта stacking - на 891 строке трейна stacking-мета-модели не хватает данных, чтобы выучить веса лучше равного голосования
- DNN (MLP с Embedding-слоем для категорий) сама по себе слабее бустингов и не помогает ансамблю - только размывает голосование
- ручной feature engineering (взаимодействия фичей вроде Sex×Pclass) не дал прироста ни деревьям (сами находят взаимодействия), ни линейным моделям - сигнал в данных исчерпывается уже имеющимися Title/Sex/Pclass
- дальнейший рост, скорее всего, упирается в потолок сигнала при 891 строке трейна, а не в выбор модели или архитектуры
