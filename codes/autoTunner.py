import optuna
import subprocess
import argparse
import os
import random
from itertools import product
from optuna.logging import get_logger, set_verbosity
import logging

# 设置 Optuna 日志级别为 INFO，以显示进度条
set_verbosity(logging.INFO)
logger = get_logger("optuna")

# 定义超参数搜索空间
HYPERPARAM_SPACE = {
    "batch_size": [8, 16, 32],
    "lr": [1e-4, 5e-4, 1e-3, 5e-3],
    "accumulation_steps": [1, 2, 4, 8],
    "epochs": [70, 100, 150, 200],
}

def generate_unique_params(n_trials):
    """
    生成唯一的参数组合。
    :param n_trials: 试验次数
    :return: 唯一的参数组合列表
    """
    all_params = list(product(
        HYPERPARAM_SPACE["batch_size"],
        HYPERPARAM_SPACE["lr"],
        HYPERPARAM_SPACE["accumulation_steps"],
        HYPERPARAM_SPACE["epochs"]
    ))
    if n_trials > len(all_params):
        raise ValueError(f"Requested {n_trials} trials, but only {len(all_params)} unique combinations exist.")
    return random.sample(all_params, n_trials)

def objective(trial, model_type, optimize_target):
    """
    训练模型并返回验证集指标。
    :param trial: Optuna 试验对象
    :param model_type: 模型类型（model0, model1, model2）
    :param optimize_target: 优化目标（accuracy 或 f1）
    :return: 验证集指标
    """
    # 定义需要调参的超参数
    batch_size = trial.suggest_categorical("batch_size", HYPERPARAM_SPACE["batch_size"])
    lr = trial.suggest_categorical("lr", HYPERPARAM_SPACE["lr"])
    accumulation_steps = trial.suggest_categorical("accumulation_steps", HYPERPARAM_SPACE["accumulation_steps"])
    epochs = trial.suggest_categorical("epochs", HYPERPARAM_SPACE["epochs"])

    # 获取 train.py 的绝对路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    train_script_path = os.path.join(current_dir, "train.py")

    # 调用 train.py 进行训练
    command = [
        "python", train_script_path,
        "--batch_size", str(batch_size),
        "--epochs", str(epochs),
        "--accumulation_steps", str(accumulation_steps),
        "--lr", str(lr),
        "--val_ratio", "0.2",
        "--disable_save",
        "--model_type", model_type  # 传递模型类型参数
    ]
    result = subprocess.run(command, capture_output=True, text=True)

    # 检查训练是否成功
    if result.returncode != 0:
        raise RuntimeError(f"Training failed with return code {result.returncode}. Error: {result.stderr}")

    # 从输出中提取验证集指标
    val_accuracies = []
    val_f1_scores = []
    for line in result.stdout.split("\n"):
        if "Val Accuracy" in line:
            val_accuracy_str = line.split("Val Accuracy: ")[1].split("%")[0].strip()
            val_accuracy = float(val_accuracy_str)
            val_accuracies.append(val_accuracy)
        if "Val F1" in line:
            val_f1_str = line.split("Val F1: ")[1].strip()
            val_f1 = float(val_f1_str)
            val_f1_scores.append(val_f1)

    if optimize_target == "accuracy":
        if not val_accuracies:
            raise ValueError("Validation accuracy not found in output.")
        return -max(val_accuracies)  # 最小化目标值
    elif optimize_target == "f1":
        if not val_f1_scores:
            raise ValueError("Validation F1 score not found in output.")
        return -max(val_f1_scores)  # 最小化目标值
    else:
        raise ValueError("Invalid optimize_target. Choose 'accuracy' or 'f1'.")

def tune_model(model_type, optimize_target, n_trials, n_jobs):
    """
    调优单个模型。
    :param model_type: 模型类型（model0, model1, model2）
    :param optimize_target: 优化目标（accuracy 或 f1）
    :param n_trials: 试验次数
    :param n_jobs: 并行任务数
    """
    # 创建 Optuna 学习任务（单目标优化）
    pruner = optuna.pruners.HyperbandPruner(
        min_resource=10,
        max_resource=100,
        reduction_factor=2
    )
    study = optuna.create_study(direction="minimize", pruner=pruner)  # 单目标优化，最小化目标值

    # 使用多线程优化
    study.optimize(lambda trial: objective(trial, model_type, optimize_target), n_trials=n_trials, n_jobs=n_jobs)

    # 输出最佳超参数
    print(f"\nBest trial for {model_type}:")
    print(f"  Value (Validation {optimize_target.capitalize()}): {-study.best_trial.value}")
    print("  Params: ")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")

def main():
    """
    主函数，用于自动调优模型超参数。
    """
    parser = argparse.ArgumentParser(description="Auto-tuner for hyperparameters.")
    parser.add_argument("--n_trials", type=int, default=50, help="Number of trials for optimization.")
    parser.add_argument("--n_jobs", type=int, default=8, help="Number of parallel jobs for optimization.")
    parser.add_argument("--model_type", type=str, default="model0", choices=["model0", "model1", "model2"], help="Type of model to use (model0, model1, model2)")
    parser.add_argument("--optimize_target", type=str, default="f1", choices=["accuracy", "f1"], help="Optimization target (accuracy or f1)")
    parser.add_argument("--all_models", action="store_true", help="Tune all models (model0, model1, model2)")
    args = parser.parse_args()

    if args.all_models:
        # 调优所有模型
        for model_type in ["model0", "model1", "model2"]:
            print(f"\n{'='*30}\nTuning {model_type}\n{'='*30}")
            tune_model(model_type, args.optimize_target, args.n_trials, args.n_jobs)
    else:
        # 调优单个模型
        tune_model(args.model_type, args.optimize_target, args.n_trials, args.n_jobs)

if __name__ == "__main__":
    main()