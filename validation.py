import copy

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import ParameterGrid, StratifiedKFold


def cross_validate_model(model, X, y, n_splits=5, random_state=42):
    """Считает accuracy модели по StratifiedKFold, возвращает массив скоров по фолдам.
    Копирует модель через copy.deepcopy вместо sklearn.base.clone: clone сравнивает параметры
    по identity после реконструкции объекта, а у CatBoost с cat_features это падает с ошибкой
    (конструктор делает копию списка cat_features), deepcopy такой проверки не делает"""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores = []

    for train_idx, val_idx in cv.split(X, y):
        fold_model = copy.deepcopy(model)
        fold_model.fit(X.iloc[train_idx], y.iloc[train_idx])
        y_pred = fold_model.predict(X.iloc[val_idx])
        scores.append(accuracy_score(y.iloc[val_idx], y_pred))

    return np.array(scores)


def evaluate_model(name, model, X, y, results, n_splits=5, random_state=42):
    """Прогоняет модель через cross_validate_model, печатает mean +- std и сохраняет scores в results[name]"""
    scores = cross_validate_model(model, X, y, n_splits=n_splits, random_state=random_state)
    results[name] = scores
    print(f"{name}: {scores.mean():.5f} +- {scores.std():.5f}")
    return scores


def tune_hyperparameters(name, model_ctor, param_grid, X, y, results, n_splits=5, random_state=42):
    """Перебирает все комбинации параметров из param_grid (dict или список dict-ов,
    формат как в sklearn ParameterGrid), для каждой считает CV-скор через cross_validate_model.
    В results[name] сохраняет массив CV-скоров лучшей комбинации (тот же формат, что у evaluate_model,
    чтобы результаты разных моделей можно было сравнивать одинаково). Возвращает лучшие параметры и этот массив"""
    best_scores = None
    best_params = None

    for params in ParameterGrid(param_grid):
        model = model_ctor(**params)
        scores = cross_validate_model(model, X, y, n_splits=n_splits, random_state=random_state)
        if best_scores is None or scores.mean() > best_scores.mean():
            best_scores = scores
            best_params = params

    results[name] = best_scores
    print(f"{name}: лучший скор {best_scores.mean():.5f} +- {best_scores.std():.5f}, параметры {best_params}")
    return best_params, best_scores


def train_kfold_and_predict(model, X, y, X_test, n_splits=5, random_state=42):
    """Обучает по одной модели на train-части каждого фолда и усредняет предсказанные
    вероятности класса 1 на X_test, используется для итогового сабмита"""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    test_proba = np.zeros(len(X_test))

    for train_idx, _ in cv.split(X, y):
        # deepcopy делает чистую копию модели с теми же гиперпараметрами, но без обучения.
        # Без этого все фолды дообучали бы один и тот же объект model,
        # и для некоторых моделей (например, с warm_start=True) это испортило бы независимость фолдов.
        fold_model = copy.deepcopy(model)
        fold_model.fit(X.iloc[train_idx], y.iloc[train_idx])
        test_proba += fold_model.predict_proba(X_test)[:, 1] / n_splits

    return (test_proba >= 0.5).astype(int)
