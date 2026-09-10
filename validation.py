import numpy as np
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_score


def cross_validate_model(model, X, y, n_splits=5, random_state=42):
    """Считает accuracy модели по StratifiedKFold, возвращает массив скоров по фолдам"""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return cross_val_score(model, X, y, cv=cv, scoring='accuracy')


def evaluate_model(name, model, X, y, results, n_splits=5, random_state=42):
    """Прогоняет модель через cross_validate_model, печатает mean +- std и сохраняет scores в results[name]"""
    scores = cross_validate_model(model, X, y, n_splits=n_splits, random_state=random_state)
    results[name] = scores
    print(f"{name}: {scores.mean():.5f} +- {scores.std():.5f}")
    return scores


def train_kfold_and_predict(model, X, y, X_test, n_splits=5, random_state=42):
    """Обучает по одной модели на train-части каждого фолда и усредняет предсказанные
    вероятности класса 1 на X_test — используется для итогового сабмита"""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    test_proba = np.zeros(len(X_test))

    for train_idx, _ in cv.split(X, y):
        # clone делает чистую копию модели с теми же гиперпараметрами, но без обучения.
        # Без этого все фолды дообучали бы один и тот же объект model,
        # и для некоторых моделей (например, с warm_start=True) это испортило бы независимость фолдов.
        fold_model = clone(model)
        fold_model.fit(X.iloc[train_idx], y.iloc[train_idx])
        test_proba += fold_model.predict_proba(X_test)[:, 1] / n_splits

    return (test_proba >= 0.5).astype(int)
