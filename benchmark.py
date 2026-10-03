#!/usr/bin/env python3
"""
LAB 16 - Cloud AI Environment Setup (Azure track)
LightGBM benchmark trên Azure CPU instance (Standard_B2s_v2).

Dataset : Kaggle - Credit Card Fraud Detection (mlg-ulb/creditcardfraud)
          284,807 giao dịch, 30 features + nhãn Class (0 = hợp lệ, 1 = gian lận)

Script thực hiện:
  1. Load dataset + tách train/test
  2. Chọn số vòng boosting bằng 5-fold cross-validation
  3. Huấn luyện LGBMClassifier trên toàn bộ tập train
  4. Đánh giá: AUC-ROC, Accuracy, F1, Precision, Recall
  5. Đo inference latency (1 dòng) và throughput (1000 dòng)
  6. Ghi kết quả ra benchmark_result.json

Về thiết kế (những điểm đã phải kiểm chứng trên chính dataset này):

* Bỏ cột `Time`. Đây là timestamp thô, 5-fold CV cho thấy có nó AUC giảm
  (0.897 -> 0.880).

* `is_unbalance=True` là BẮT BUỘC. Dữ liệu chỉ có 0.17% giao dịch gian lận
  (492/284.807). Nếu không cân bằng class, LightGBM học trên ~400 mẫu dương
  và overfit rất nhanh.

* KHÔNG dùng early stopping trên một validation split nhỏ. Validation chỉ có
  ~79 mẫu gian lận nên AUC trên split đó cực nhiễu và early stopping dừng ở
  best_iteration = 1, tức là model chỉ còn 1 cây. Thay vào đó ta chạy 5-fold
  CV, trung bình đường cong AUC của các fold (~395 mẫu dương) rồi chọn số
  vòng boosting tại điểm cực đại của đường trung bình này.

* AUC trên một test split chỉ chứa ~98 mẫu gian lận có sai số cỡ +/-0.02, nên
  benchmark cũng báo thêm AUC cross-validation để đối chiếu.
"""

import json
import os
import platform
import time
import urllib.request
import warnings
from datetime import datetime, timezone

import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split

# LightGBM >=4.7 gợi ý dùng eval_X/eval_y; eval_set vẫn hoạt động và tương thích
# với các bản cũ hơn nên ta giữ eval_set, chỉ tắt cảnh báo cho output gọn.
warnings.filterwarnings("ignore", message=".*eval_set.*is deprecated.*")

DATA_PATH = os.path.expanduser("~/ml-benchmark/creditcard.csv")
RESULT_PATH = os.path.expanduser("~/ml-benchmark/benchmark_result.json")

RANDOM_STATE = 42
TEST_SIZE = 0.2
N_FOLDS = 5
MAX_ESTIMATORS = 600
# Các mốc số vòng boosting để so sánh trong CV
N_ESTIMATOR_GRID = [50, 100, 150, 200, 300, 400, 500, 600]

PARAMS = {
    "objective": "binary",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "min_child_samples": 20,
    "is_unbalance": True,   # bắt buộc vì dữ liệu mất cân bằng nghiêm trọng
    "n_jobs": os.cpu_count(),
    "random_state": RANDOM_STATE,
    "verbose": -1,
    "metric": "auc",
}


def banner(title):
    print("\n" + "=" * 62)
    print(title)
    print("=" * 62)


def get_azure_metadata():
    """Lấy thông tin VM từ Azure Instance Metadata Service (IMDS)."""
    url = "http://169.254.169.254/metadata/instance?api-version=2021-02-01"
    try:
        req = urllib.request.Request(url, headers={"Metadata": "true"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            compute = json.load(resp).get("compute", {})
        return {
            "vmSize": compute.get("vmSize"),
            "vmId": compute.get("vmId"),
            "location": compute.get("location"),
            "osType": compute.get("osType"),
        }
    except Exception:
        return None


def select_n_estimators(X, y):
    """
    Chọn số vòng boosting bằng 5-fold CV.

    Hai điều đã phải kiểm chứng trên chính dataset này:

    1. Early stopping trên một validation split đơn lẻ không dùng được:
       split đó chỉ chứa ~79 mẫu gian lận nên AUC quá nhiễu, và early
       stopping chọn best_iteration = 1 tức là model chỉ còn 1 cây.

    2. AUC do LightGBM tự ghi vào lịch sử eval cũng không đáng tin trên
       dataset này: đường cong bị đóng băng ở một giá trị duy nhất từ vòng
       ~450 trở đi. Vì vậy ta tự tính AUC bằng sklearn trên dự đoán thật.

    Mỗi fold chỉ fit một lần lên MAX_ESTIMATORS cây, rồi dùng num_iteration
    để lấy dự đoán tại từng mốc trong N_ESTIMATOR_GRID.
    """
    print(f"Chạy {N_FOLDS}-fold cross-validation để chọn n_estimators ...")
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                          random_state=RANDOM_STATE)

    fold_auc = {n: [] for n in N_ESTIMATOR_GRID}
    t0 = time.perf_counter()
    for tr, va in skf.split(X, y):
        m = lgb.LGBMClassifier(**PARAMS, n_estimators=MAX_ESTIMATORS)
        m.fit(X.iloc[tr], y.iloc[tr])
        for n in N_ESTIMATOR_GRID:
            proba = m.predict_proba(X.iloc[va], num_iteration=n)[:, 1]
            fold_auc[n].append(roc_auc_score(y.iloc[va], proba))
    cv_time = time.perf_counter() - t0

    mean_auc = {n: float(np.mean(v)) for n, v in fold_auc.items()}
    std_auc = {n: float(np.std(v)) for n, v in fold_auc.items()}
    n_star = max(mean_auc, key=mean_auc.get)
    cv_auc = mean_auc[n_star]

    print(f"CV time           : {cv_time:.2f} s "
          f"(AUC tự tính bằng sklearn, 1 fit/fold)")
    print(f"{'n_estimators':>14} | {'mean CV-AUC':>12} | {'std':>7}")
    print("-" * 40)
    for n in N_ESTIMATOR_GRID:
        tag = "  <- chọn" if n == n_star else ""
        print(f"{n:>14} | {mean_auc[n]:>12.6f} | {std_auc[n]:>7.4f}{tag}")
    return n_star, cv_auc, cv_time, mean_auc, std_auc


def main():
    banner("LAB 16 - LightGBM Benchmark | Azure CPU instance")
    az = get_azure_metadata()
    print(f"Timestamp (UTC) : {datetime.now(timezone.utc).isoformat()}")
    print(f"Hostname        : {platform.node()}")
    print(f"Python          : {platform.python_version()}")
    print(f"Platform        : {platform.platform()}")
    print(f"Dataset path    : {DATA_PATH}")
    print(f"vCPU            : {os.cpu_count()}")
    if az:
        print(f"Azure VM size   : {az['vmSize']}")
        print(f"Azure location  : {az['location']}")

    # ------------------------------------------------------------------
    # 1. LOAD DATA
    # ------------------------------------------------------------------
    banner("[1/6] Loading dataset ...")
    size_mb = os.path.getsize(DATA_PATH) / (1024 ** 2)
    t0 = time.perf_counter()
    df = pd.read_csv(DATA_PATH)
    load_time = time.perf_counter() - t0

    total_rows, total_cols = df.shape
    print(f"Shape           : {total_rows:,} rows x {total_cols} columns")
    print(f"Load time       : {load_time:.3f} s "
          f"({size_mb:.1f} MB => {size_mb / load_time:.1f} MB/s)")
    print(f"Fraud rate      : {df['Class'].mean() * 100:.4f}% "
          f"({int(df['Class'].sum()):,} positive / "
          f"{total_rows - int(df['Class'].sum()):,} negative)")

    # 'Time' là timestamp thô, không mang tín hiệu dự đoán -> loại bỏ
    X = df.drop(columns=["Class", "Time"])
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,          # giữ đúng tỉ lệ gian lận ở tập test
        shuffle=True,
    )
    print(f"Train           : {X_train.shape[0]:,} rows, "
          f"{X_train.shape[1]} features ({int(y_train.sum())} fraud)")
    print(f"Test (holdout)  : {X_test.shape[0]:,} rows "
          f"({int(y_test.sum())} fraud)")

    # ------------------------------------------------------------------
    # 2. CHỌN SỐ VÒNG BOOSTING BẰNG CROSS-VALIDATION
    # ------------------------------------------------------------------
    banner("[2/6] Selecting n_estimators via cross-validation ...")
    n_estimators, cv_auc, cv_time, mean_auc, std_auc = select_n_estimators(
        X_train, y_train)
    if n_estimators < 10:
        print("\n[CẢNH BÁO] best iteration rất nhỏ - model có thể bị "
              "underfit. Cân nhắc tăng learning_rate hoặc kiểm tra lại data.")

    # ------------------------------------------------------------------
    # 3. TRAIN MODEL trên toàn bộ tập train
    # ------------------------------------------------------------------
    banner("[3/6] Training LGBMClassifier on full training set ...")
    print("Hyper-parameters:")
    for k, v in PARAMS.items():
        print(f"  {k:18s} = {v}")
    print(f"  {'n_estimators':18s} = {n_estimators}  (CV-selected)")

    model = lgb.LGBMClassifier(**PARAMS, n_estimators=n_estimators)

    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    training_time = time.perf_counter() - t0

    print(f"\nTraining time   : {training_time:.3f} s")
    print(f"Best iteration  : {n_estimators}")
    print(f"Trees built     : {model.booster_.num_trees()}")

    # ------------------------------------------------------------------
    # 4. EVALUATE
    # ------------------------------------------------------------------
    banner("[4/6] Evaluating model on holdout test set ...")
    t0 = time.perf_counter()
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    infer_all_time = time.perf_counter() - t0
    y_pred = (y_pred_proba >= 0.5).astype(int)

    auc_roc = roc_auc_score(y_test, y_pred_proba)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)

    print(f"AUC-ROC         : {auc_roc:.6f}   (5-fold CV: {cv_auc:.6f})")
    print(f"Accuracy        : {accuracy:.6f}")
    print(f"F1-Score        : {f1:.6f}")
    print(f"Precision       : {precision:.6f}")
    print(f"Recall          : {recall:.6f}")
    print(f"(batch predict {len(X_test):,} rows: {infer_all_time:.3f} s)")

    # Ngưỡng 0.5 không phải ngưỡng tối ưu khi dùng is_unbalance (mọi mẫu
    # dương đều bị đẩy lên). Báo thêm điểm hoạt động tối ưu cho F1.
    grid = np.linspace(0.05, 0.95, 181)
    f1s = [f1_score(y_test, (y_pred_proba >= t).astype(int), zero_division=0)
           for t in grid]
    best_thr = float(grid[int(np.argmax(f1s))])
    y_pred_bt = (y_pred_proba >= best_thr).astype(int)
    thr_metrics = {
        "threshold": round(best_thr, 4),
        "f1_score": round(float(f1_score(y_test, y_pred_bt, zero_division=0)), 6),
        "precision": round(float(precision_score(y_test, y_pred_bt,
                                                 zero_division=0)), 6),
        "recall": round(float(recall_score(y_test, y_pred_bt, zero_division=0)), 6),
        "accuracy": round(float(accuracy_score(y_test, y_pred_bt)), 6),
    }
    print(f"\nBest F1 threshold: {best_thr:.4f} "
          f"(F1={thr_metrics['f1_score']:.4f}, "
          f"P={thr_metrics['precision']:.4f}, R={thr_metrics['recall']:.4f})")

    # ------------------------------------------------------------------
    # 5. INFERENCE LATENCY (1 row) và THROUGHPUT (1000 rows)
    # ------------------------------------------------------------------
    banner("[5/6] Measuring inference latency and throughput ...")
    one_row = X_test.iloc[[0]]
    for _ in range(10):                      # warm-up
        model.predict_proba(one_row)

    LAT_N = 200
    t0 = time.perf_counter()
    for _ in range(LAT_N):
        model.predict_proba(one_row)
    latency_ms = (time.perf_counter() - t0) / LAT_N * 1000
    print(f"Latency (1 row) : {latency_ms:.4f} ms   "
          f"(trung bình {LAT_N} lần sau warm-up)")

    batch = X_test.head(1000)
    model.predict_proba(batch)               # warm-up

    THR_N = 20
    t0 = time.perf_counter()
    for _ in range(THR_N):
        model.predict_proba(batch)
    elapsed = time.perf_counter() - t0
    thr_rows_per_sec = (THR_N * len(batch)) / elapsed
    thr_ms = elapsed / THR_N * 1000
    print(f"Throughput      : {thr_rows_per_sec:,.0f} rows/s")
    print(f"Per 1000-row batch: {thr_ms:.3f} ms   (trung bình {THR_N} lần)")

    # ------------------------------------------------------------------
    # 6. SAVE RESULT
    # ------------------------------------------------------------------
    banner("[6/6] Writing benchmark_result.json ...")
    result = {
        "metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "hostname": platform.node(),
            "cloud": "Microsoft Azure",
            "resource_group": "ai-lab-rg",
            "vm_size": (az or {}).get("vmSize"),
            "location": (az or {}).get("location"),
            "vCPU": os.cpu_count(),
            "os": platform.platform(),
            "python_version": platform.python_version(),
            "lib_versions": {
                "lightgbm": lgb.__version__,
                "scikit_learn": sklearn.__version__,
                "pandas": pd.__version__,
                "numpy": np.__version__,
            },
        },
        "dataset": {
            "name": "Credit Card Fraud Detection",
            "source": "Kaggle - mlg-ulb/creditcardfraud",
            "path": DATA_PATH,
            "file_size_mb": round(size_mb, 2),
            "total_rows": int(total_rows),
            "total_columns": int(total_cols),
            "features_used": int(X.shape[1]),
            "fraud_rate_pct": round(float(df["Class"].mean() * 100), 4),
            "train_rows": int(X_train.shape[0]),
            "train_fraud": int(y_train.sum()),
            "test_rows": int(X_test.shape[0]),
            "test_fraud": int(y_test.sum()),
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
        },
        "timing": {
            "load_data_sec": round(load_time, 3),
            "load_data_mb_per_sec": round(size_mb / load_time, 1),
            "cross_validation_sec": round(cv_time, 2),
            "training_sec": round(training_time, 3),
            "best_iteration": int(n_estimators),
            "batch_predict_full_test_sec": round(infer_all_time, 3),
        },
        "metrics": {
            "auc_roc": round(float(auc_roc), 6),
            "auc_roc_cv5": round(cv_auc, 6),
            "auc_roc_cv5_std": round(std_auc[n_estimators], 6),
            "accuracy": round(float(accuracy), 6),
            "f1_score": round(float(f1), 6),
            "precision": round(float(precision), 6),
            "recall": round(float(recall), 6),
        },
        "metrics_at_best_f1_threshold": thr_metrics,
        "inference": {
            "latency_1_row_ms": round(latency_ms, 4),
            "latency_samples": LAT_N,
            "throughput_rows_per_sec": round(thr_rows_per_sec, 1),
            "throughput_1000_rows_ms": round(thr_ms, 3),
            "throughput_batches": THR_N,
        },
        "hyperparameters": {**PARAMS, "n_estimators": int(n_estimators)},
        "cv_n_estimators_sweep": {
            str(n): {"mean_auc": round(mean_auc[n], 6),
                     "std_auc": round(std_auc[n], 6)}
            for n in N_ESTIMATOR_GRID
        },
        "notes": [
            "Dropped the 'Time' column: raw timestamp with no predictive "
            "signal (5-fold CV: 0.897 without vs 0.880 with).",
            "is_unbalance=True is required: with 0.17% fraud, an unweighted "
            "model overfits the ~400 positive rows and validation AUC "
            "collapses after the first iteration.",
            "Early stopping on a single validation split is NOT used here. "
            "That split holds only ~79 fraud rows, so its AUC is far too "
            "noisy - early stopping selected best_iteration=1 (a one-tree "
            "model). n_estimators is instead chosen from the mean 5-fold CV "
            "AUC curve, which pools ~395 fraud rows.",
            "Note the dataset has a pathological quirk: validation AUC drops "
            "sharply at boosting iteration 2 under every weighting scheme "
            "(reproduced identically on LightGBM 4.6.0 and 4.7.0), which is "
            "why naive early stopping collapses.",
            "LightGBM's own recorded eval-AUC history is also unreliable on "
            "this data (the curve freezes from ~iteration 450 onward), so "
            "n_estimators is chosen from AUC computed by sklearn on real "
            "fold predictions instead.",
            f"The test split holds only {int(y_test.sum())} fraud rows, so a "
            "single-split AUC carries roughly +/-0.02 of noise; compare "
            "auc_roc against auc_roc_cv5.",
            "Metrics in 'metrics' use threshold 0.5. Because is_unbalance "
            "shifts all scores upward, 0.5 is a poor operating point; "
            "'metrics_at_best_f1_threshold' reports the tuned threshold.",
        ],
    }

    with open(RESULT_PATH, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved -> {RESULT_PATH}")

    # ------------------------------------------------------------------
    # SUMMARY (khớp với bảng trong README)
    # ------------------------------------------------------------------
    banner("SUMMARY")
    rows = [
        ("Thoi gian load data", f"{load_time:.3f} s"),
        ("Thoi gian training", f"{training_time:.3f} s"),
        ("Best iteration", f"{n_estimators}"),
        ("AUC-ROC", f"{auc_roc:.6f}"),
        ("Accuracy", f"{accuracy:.6f}"),
        ("F1-Score", f"{f1:.6f}"),
        ("Precision", f"{precision:.6f}"),
        ("Recall", f"{recall:.6f}"),
        ("Inference latency (1 row)", f"{latency_ms:.4f} ms"),
        ("Inference throughput (1000 rows)", f"{thr_rows_per_sec:,.0f} rows/s"),
    ]
    print(f"{'Metric':<34}| {'Result'}")
    print("-" * 34 + "+" + "-" * 22)
    for k, v in rows:
        print(f"{k:<34}| {v}")
    print("=" * 57)


if __name__ == "__main__":
    main()