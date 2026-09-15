from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from dnn import EmbeddingMLPClassifier

MODEL_CLASSES = {
    'logreg': LogisticRegression,
    'knn': KNeighborsClassifier,
    'decision_tree': DecisionTreeClassifier,
    'random_forest': RandomForestClassifier,
    'xgboost': XGBClassifier,
    'lightgbm': LGBMClassifier,
    'catboost': CatBoostClassifier,
    'dnn_embedding': EmbeddingMLPClassifier,
}

FINAL_ESTIMATOR_CLASSES = {
    'logreg_noreg': LogisticRegression,
    'ridge': RidgeClassifier,
}


def build_base_model(name, cfg):
    """Строит одну базовую модель по имени из cfg.models с зафиксированными гиперпараметрами"""
    params = dict(cfg.models[name].params)
    return MODEL_CLASSES[name](**params)


def build_model(name, cfg):
    """Строит модель или ансамбль по имени: сначала ищет одиночную модель в cfg.models,
    иначе собирает ансамбль (voting/stacking) из cfg.ensembles по описанным в нём base_models"""
    if name in cfg.models:
        return build_base_model(name, cfg)

    ensemble_cfg = cfg.ensembles[name]
    base_estimators = [(base_name, build_base_model(base_name, cfg)) for base_name in ensemble_cfg.base_models]

    if ensemble_cfg.type == 'voting':
        return VotingClassifier(estimators=base_estimators, voting=ensemble_cfg.voting, n_jobs=-1)

    if ensemble_cfg.type == 'stacking':
        final_params = dict(cfg.final_estimators[ensemble_cfg.final_estimator].params)
        final_estimator = FINAL_ESTIMATOR_CLASSES[ensemble_cfg.final_estimator](**final_params)
        return StackingClassifier(
            estimators=base_estimators, final_estimator=final_estimator, cv=ensemble_cfg.cv, n_jobs=-1
        )

    raise ValueError(f"Неизвестный тип ансамбля: {ensemble_cfg.type}")
