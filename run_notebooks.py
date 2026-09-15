import subprocess
import sys
import time

NOTEBOOKS = [
    'eda_and_hypotheses.ipynb',
    'simple_baseline_and_experimentations.ipynb',
    'classic_models_exploration.ipynb',
    'DNN_exploration.ipynb',
    'ensembles.ipynb',
    'optuna_exploration.ipynb',
]


def run_notebook(path):
    """Выполняет один ноутбук на месте (--inplace) тем же интерпретатором, что запустил скрипт"""
    result = subprocess.run(
        [sys.executable, '-m', 'jupyter', 'nbconvert', '--to', 'notebook', '--execute', '--inplace', path],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(result.stderr[-3000:])
    return result.returncode == 0


def main():
    failed = []
    for path in NOTEBOOKS:
        print(f"Запускаю {path}...")
        start = time.time()
        ok = run_notebook(path)
        elapsed = time.time() - start
        print(f"{path}: {'OK' if ok else 'FAILED'} ({elapsed:.0f}s)")
        if not ok:
            failed.append(path)

    if failed:
        print(f"\nНе выполнились: {', '.join(failed)}")
        sys.exit(1)
    print("\nВсе ноутбуки выполнены успешно")


if __name__ == '__main__':
    main()
