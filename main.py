import argparse

import pandas as pd
from omegaconf import OmegaConf

from preprocessing import build_features
from validation import cross_validate_model, train_kfold_and_predict
from models import build_model


def evaluate_all(cfg, X_train, y_train):
    """model: all - прогоняет по CV каждую модель и каждый ансамбль из конфига, печатает
    сравнительную таблицу (аналог итоговой таблицы из ensembles.ipynb). Без сабмита -
    для сабмита нужно выбрать одну конкретную модель в model"""
    results = {}
    for name in list(cfg.models.keys()) + list(cfg.ensembles.keys()):
        model = build_model(name, cfg)
        scores = cross_validate_model(
            model, X_train, y_train, n_splits=cfg.validation.n_splits, random_state=cfg.validation.random_state
        )
        results[name] = scores
        print(f"{name}: {scores.mean():.5f} +- {scores.std():.5f}")

    summary = pd.DataFrame({
        'mean': {name: scores.mean() for name, scores in results.items()},
        'std': {name: scores.std() for name, scores in results.items()},
    }).sort_values('mean', ascending=False)
    print("\n" + summary.to_string())


def main(config_path):
    cfg = OmegaConf.load(config_path)

    train_df = pd.read_csv(cfg.data.train_path)
    test_df = pd.read_csv(cfg.data.test_path)
    X_train, y_train, X_test, test_passenger_ids = build_features(train_df, test_df)

    if cfg.model == 'all':
        evaluate_all(cfg, X_train, y_train)
        return

    model = build_model(cfg.model, cfg)
    scores = cross_validate_model(
        model, X_train, y_train, n_splits=cfg.validation.n_splits, random_state=cfg.validation.random_state
    )
    print(f"{cfg.model}: CV accuracy {scores.mean():.5f} +- {scores.std():.5f}")

    predictions = train_kfold_and_predict(
        model, X_train, y_train, X_test, n_splits=cfg.validation.n_splits, random_state=cfg.validation.random_state
    )
    submission = pd.DataFrame({'PassengerId': test_passenger_ids, 'Survived': predictions})
    submission.to_csv(cfg.data.submission_path, index=False)
    print(f"Сабмит сохранён в {cfg.data.submission_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml')
    args = parser.parse_args()
    main(args.config)
