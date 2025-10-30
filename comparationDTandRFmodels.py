import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os

from joblib import dump
from sklearn.model_selection import train_test_split,learning_curve,cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report, brier_score_loss
)
from scipy.sparse import vstack
from sklearn.calibration import calibration_curve



# 1. Dataset Preprocessing
# 1.1 get data from bank-full file
df = pd.read_csv("bank-full.csv", sep=";")
X = df.drop("y", axis=1)
y = df["y"].map({"no": 0, "yes": 1})  # 转换为二分类 0/1

# 1.2 split dataset into training(60%), validation(20%) and test(20%) sets.
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
)

features_without_duration = [col for col in X.columns if col != 'duration']
X_train_no_dur = X_train[features_without_duration]
X_val_no_dur = X_val[features_without_duration]
X_test_no_dur = X_test[features_without_duration]


# 1.3 One-Hot encoding for categorical features
categorical_features = X.select_dtypes(include=["object"]).columns
preprocessor = ColumnTransformer(
    transformers=[
        ("onehot", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ],
    remainder="passthrough"
)
preprocessor_no_dur = ColumnTransformer(
    transformers=[
        ("onehot", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ],
    remainder="passthrough"
)
preprocessor.fit(X_train) # fit in training sets only
X_train_enc = preprocessor.transform(X_train)
X_val_enc   = preprocessor.transform(X_val)
X_test_enc  = preprocessor.transform(X_test)

preprocessor_no_dur.fit(X_train_no_dur) # fit in training sets only
X_train_no_dur_enc = preprocessor_no_dur.transform(X_train_no_dur)
X_val_no_dur_enc = preprocessor_no_dur.transform(X_val_no_dur)
X_test_no_dur_enc = preprocessor_no_dur.transform(X_test_no_dur)

# 2. Model Training and Hyperparameter Tuning
# ---- Decision Tree Hyperparameter Tuning ----
dt_params = {"max_depth": [5, 10, 15, 20, 25, 30, None],
             "min_samples_split": [2, 5, 10, 15, 20, 25, 30]}

results_dt = []
best_dt_score, best_dt_params = 0, None

for depth in dt_params["max_depth"]:
    for min_split in dt_params["min_samples_split"]:
        dt = DecisionTreeClassifier(max_depth=depth,
                                    min_samples_split=min_split,
                                    random_state=42)
        dt.fit(X_train_enc, y_train)
        y_val_pred = dt.predict(X_val_enc)
        f1 = f1_score(y_val, y_val_pred)
        results_dt.append({"max_depth": str(depth), "min_samples_split": min_split, "f1": f1})
        if f1 > best_dt_score:
            best_dt_score = f1
            best_dt_params = {"max_depth": depth, "min_samples_split": min_split}

print("Decision Tree Best Parameters:", best_dt_params)

# Convert to DataFrame and then display as a heatmap matrix
df_dt_heat = pd.DataFrame(results_dt)
heatmap_dt = df_dt_heat.pivot(index="max_depth", columns="min_samples_split", values="f1")

plt.figure(figsize=(8,6))
sns.heatmap(heatmap_dt, annot=True, fmt=".3f", cmap="YlGnBu")
plt.title("Decision Tree: F1 Score by (max_depth, min_samples_split)")
plt.ylabel("max_depth")
plt.xlabel("min_samples_split")
plt.show()

# ---- Random Forest Hyperparameter Tuning ----
rf_params = {"n_estimators": [10, 50, 100, 200, 250, 300],
             "max_depth": [5, 10, 15, 20, 25, None]}

results_rf = []
best_rf_score, best_rf_params = 0, None

for n in rf_params["n_estimators"]:
    for depth in rf_params["max_depth"]:
        rf = RandomForestClassifier(n_estimators=n,
                                    max_depth=depth,
                                    random_state=42,
                                    n_jobs=-1)
        rf.fit(X_train_enc, y_train)
        y_val_pred = rf.predict(X_val_enc)
        f1 = f1_score(y_val, y_val_pred)
        results_rf.append({"n_estimators": n, "max_depth": str(depth), "f1": f1})
        if f1 > best_rf_score:
            best_rf_score = f1
            best_rf_params = {"n_estimators": n, "max_depth": depth}

print("Random Forest Best Parameters:", best_rf_params)

# Convert to DataFrame and then display as a heatmap matrix
df_rf_heat = pd.DataFrame(results_rf)
heatmap_rf = df_rf_heat.pivot(index="max_depth", columns="n_estimators", values="f1")

plt.figure(figsize=(8,6))
sns.heatmap(heatmap_rf, annot=True, fmt=".3f", cmap="YlOrRd")
plt.title("Random Forest: F1 Score by (n_estimators, max_depth)")
plt.ylabel("max_depth")
plt.xlabel("n_estimators")
plt.show()

# 3. Retraining with Best Parameters
X_train_val_enc = vstack([X_train_enc, X_val_enc])
y_train_val = np.concatenate([y_train, y_val])


X_train_val_no_dur_enc = vstack([X_train_no_dur_enc, X_val_no_dur_enc])
y_train_val_no_dur = np.concatenate([y_train, y_val])

best_dt = DecisionTreeClassifier(**best_dt_params, random_state=42)
best_dt.fit(X_train_val_enc, y_train_val)
best_dt_no_dur = DecisionTreeClassifier(**best_dt_params, random_state=42)
best_dt_no_dur.fit(X_train_val_no_dur_enc, y_train_val_no_dur)

best_rf = RandomForestClassifier(**best_rf_params, random_state=42, n_jobs=-1)
best_rf.fit(X_train_val_enc, y_train_val)
best_rf_no_dur = RandomForestClassifier(**best_rf_params, random_state=42, n_jobs=-1)
best_rf_no_dur.fit(X_train_val_no_dur_enc, y_train_val_no_dur)


models = {"Decision Tree": best_dt, "Random Forest": best_rf}
models_no_dur = {"Decision Tree no dur": best_dt_no_dur, "Random Forest no dur": best_rf_no_dur}
#--------------------------

# ============================================
# 超参数对比分析：模型大小、准确率、效率
# ============================================

def model_size_in_mb(model, filename="temp_model.joblib"):
    """计算模型文件大小（MB）"""
    dump(model, filename)
    size_mb = os.path.getsize(filename) / (1024 * 1024)
    os.remove(filename)
    return size_mb


def measure_model_metrics(model, X_train, y_train, X_test, y_test):
    """测量模型的大小、准确率和推理时间"""
    # 训练模型
    start_train = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_train

    # 测试集预测
    start_test = time.time()
    y_pred = model.predict(X_test)
    test_time = time.time() - start_test

    # 计算准确率
    acc = accuracy_score(y_test, y_pred)

    # 计算模型大小
    size_mb = model_size_in_mb(model)

    return {
        'accuracy': acc,
        'model_size_mb': size_mb,
        'train_time': train_time,
        'inference_time': test_time
    }


# ============================================
# 1. Decision Tree 超参数扫描
# ============================================
print("=" * 50)
print("Decision Tree: Hyperparameter Comparison")
print("=" * 50)

dt_results = []
dt_max_depths = [5, 10, 15, 20, 25, 30, None]

for depth in dt_max_depths:
    dt = DecisionTreeClassifier(max_depth=depth, random_state=42)
    metrics = measure_model_metrics(dt, X_train_enc, y_train, X_test_enc, y_test)

    dt_results.append({
        'max_depth': str(depth) if depth is not None else 'None',
        'accuracy': metrics['accuracy'],
        'model_size_mb': metrics['model_size_mb'],
        'train_time': metrics['train_time'],
        'inference_time': metrics['inference_time']
    })

    print(f"max_depth={depth}: Acc={metrics['accuracy']:.4f}, "
          f"Size={metrics['model_size_mb']:.3f}MB, "
          f"Train={metrics['train_time']:.3f}s, "
          f"Test={metrics['inference_time']:.4f}s")

df_dt = pd.DataFrame(dt_results)

# ============================================
# 2. Random Forest 超参数扫描
# ============================================
print("\n" + "=" * 50)
print("Random Forest: Hyperparameter Comparison")
print("=" * 50)

rf_results = []
rf_n_estimators = [10, 50, 100, 150, 200, 250, 300]

for n_est in rf_n_estimators:
    rf = RandomForestClassifier(n_estimators=n_est, random_state=42, n_jobs=-1)
    metrics = measure_model_metrics(rf, X_train_enc, y_train, X_test_enc, y_test)

    rf_results.append({
        'n_estimators': n_est,
        'accuracy': metrics['accuracy'],
        'model_size_mb': metrics['model_size_mb'],
        'train_time': metrics['train_time'],
        'inference_time': metrics['inference_time']
    })

    print(f"n_estimators={n_est}: Acc={metrics['accuracy']:.4f}, "
          f"Size={metrics['model_size_mb']:.2f}MB, "
          f"Train={metrics['train_time']:.2f}s, "
          f"Test={metrics['inference_time']:.4f}s")

df_rf = pd.DataFrame(rf_results)

# ============================================
# 3. 可视化对比
# ============================================

# 设置绘图风格
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# --- 图1：3x2 综合对比图 ---
fig, axes = plt.subplots(3, 2, figsize=(14, 12))
fig.suptitle('Hyperparameter Impact: Decision Tree vs Random Forest',
             fontsize=16, fontweight='bold')

# Decision Tree: Accuracy
axes[0, 0].plot(df_dt['max_depth'], df_dt['accuracy'], 'o-', color='steelblue', linewidth=2, markersize=8)
axes[0, 0].set_xlabel('max_depth', fontsize=11)
axes[0, 0].set_ylabel('Accuracy', fontsize=11)
axes[0, 0].set_title('Decision Tree: Accuracy vs max_depth', fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)

# Random Forest: Accuracy
axes[0, 1].plot(df_rf['n_estimators'], df_rf['accuracy'], 'o-', color='coral', linewidth=2, markersize=8)
axes[0, 1].set_xlabel('n_estimators', fontsize=11)
axes[0, 1].set_ylabel('Accuracy', fontsize=11)
axes[0, 1].set_title('Random Forest: Accuracy vs n_estimators', fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)

# Decision Tree: Model Size
axes[1, 0].plot(df_dt['max_depth'], df_dt['model_size_mb'], 's-', color='mediumseagreen', linewidth=2, markersize=8)
axes[1, 0].set_xlabel('max_depth', fontsize=11)
axes[1, 0].set_ylabel('Model Size (MB)', fontsize=11)
axes[1, 0].set_title('Decision Tree: Model Size vs max_depth', fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)

# Random Forest: Model Size
axes[1, 1].plot(df_rf['n_estimators'], df_rf['model_size_mb'], 's-', color='gold', linewidth=2, markersize=8)
axes[1, 1].set_xlabel('n_estimators', fontsize=11)
axes[1, 1].set_ylabel('Model Size (MB)', fontsize=11)
axes[1, 1].set_title('Random Forest: Model Size vs n_estimators', fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)

# Decision Tree: Inference Time
axes[2, 0].plot(df_dt['max_depth'], df_dt['inference_time'], '^-', color='mediumpurple', linewidth=2, markersize=8)
axes[2, 0].set_xlabel('max_depth', fontsize=11)
axes[2, 0].set_ylabel('Inference Time (s)', fontsize=11)
axes[2, 0].set_title('Decision Tree: Inference Time vs max_depth', fontweight='bold')
axes[2, 0].grid(True, alpha=0.3)

# Random Forest: Inference Time
axes[2, 1].plot(df_rf['n_estimators'], df_rf['inference_time'], '^-', color='hotpink', linewidth=2, markersize=8)
axes[2, 1].set_xlabel('n_estimators', fontsize=11)
axes[2, 1].set_ylabel('Inference Time (s)', fontsize=11)
axes[2, 1].set_title('Random Forest: Inference Time vs n_estimators', fontweight='bold')
axes[2, 1].grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig('hyperparameter_comparison_grid.png', dpi=300, bbox_inches='tight')
plt.show()

# --- 图2：Accuracy vs Model Size 散点图 ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Trade-off: Accuracy vs Model Size', fontsize=16, fontweight='bold')

# Decision Tree
axes[0].scatter(df_dt['model_size_mb'], df_dt['accuracy'],
                s=200, c=range(len(df_dt)), cmap='Blues',
                edgecolors='black', linewidth=1.5, alpha=0.7)
for i, row in df_dt.iterrows():
    axes[0].annotate(row['max_depth'],
                     (row['model_size_mb'], row['accuracy']),
                     fontsize=9, ha='right', va='bottom')
axes[0].set_xlabel('Model Size (MB)', fontsize=12)
axes[0].set_ylabel('Accuracy', fontsize=12)
axes[0].set_title('Decision Tree', fontweight='bold')
axes[0].grid(True, alpha=0.3)

# Random Forest
axes[1].scatter(df_rf['model_size_mb'], df_rf['accuracy'],
                s=200, c=range(len(df_rf)), cmap='Oranges',
                edgecolors='black', linewidth=1.5, alpha=0.7)
for i, row in df_rf.iterrows():
    axes[1].annotate(f"{row['n_estimators']}",
                     (row['model_size_mb'], row['accuracy']),
                     fontsize=9, ha='right', va='bottom')
axes[1].set_xlabel('Model Size (MB)', fontsize=12)
axes[1].set_ylabel('Accuracy', fontsize=12)
axes[1].set_title('Random Forest', fontweight='bold')
axes[1].grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('accuracy_vs_size_scatter.png', dpi=300, bbox_inches='tight')
plt.show()

# --- 图3：Accuracy vs Inference Time 散点图 ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Trade-off: Accuracy vs Inference Time', fontsize=16, fontweight='bold')

# Decision Tree
axes[0].scatter(df_dt['inference_time'], df_dt['accuracy'],
                s=200, c=range(len(df_dt)), cmap='Greens',
                edgecolors='black', linewidth=1.5, alpha=0.7)
for i, row in df_dt.iterrows():
    axes[0].annotate(row['max_depth'],
                     (row['inference_time'], row['accuracy']),
                     fontsize=9, ha='right', va='bottom')
axes[0].set_xlabel('Inference Time (s)', fontsize=12)
axes[0].set_ylabel('Accuracy', fontsize=12)
axes[0].set_title('Decision Tree', fontweight='bold')
axes[0].grid(True, alpha=0.3)

# Random Forest
axes[1].scatter(df_rf['inference_time'], df_rf['accuracy'],
                s=200, c=range(len(df_rf)), cmap='Reds',
                edgecolors='black', linewidth=1.5, alpha=0.7)
for i, row in df_rf.iterrows():
    axes[1].annotate(f"{row['n_estimators']}",
                     (row['inference_time'], row['accuracy']),
                     fontsize=9, ha='right', va='bottom')
axes[1].set_xlabel('Inference Time (s)', fontsize=12)
axes[1].set_ylabel('Accuracy', fontsize=12)
axes[1].set_title('Random Forest', fontweight='bold')
axes[1].grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('accuracy_vs_time_scatter.png', dpi=300, bbox_inches='tight')
plt.show()

# --- 图4：综合对比表格 ---
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Detailed Comparison Table', fontsize=16, fontweight='bold')

# Decision Tree 表格
axes[0].axis('tight')
axes[0].axis('off')
table_dt = axes[0].table(
    cellText=df_dt.round(4).values,
    colLabels=df_dt.columns,
    cellLoc='center',
    loc='center',
    colWidths=[0.15, 0.15, 0.2, 0.15, 0.2]
)
table_dt.auto_set_font_size(False)
table_dt.set_fontsize(9)
table_dt.scale(1, 2)
for i in range(len(df_dt.columns)):
    table_dt[(0, i)].set_facecolor('#4472C4')
    table_dt[(0, i)].set_text_props(weight='bold', color='white')
axes[0].set_title('Decision Tree Performance Metrics',
                  fontweight='bold', fontsize=12, pad=20)

# Random Forest 表格
axes[1].axis('tight')
axes[1].axis('off')
table_rf = axes[1].table(
    cellText=df_rf.round(4).values,
    colLabels=df_rf.columns,
    cellLoc='center',
    loc='center',
    colWidths=[0.15, 0.15, 0.2, 0.15, 0.2]
)
table_rf.auto_set_font_size(False)
table_rf.set_fontsize(9)
table_rf.scale(1, 2)
for i in range(len(df_rf.columns)):
    table_rf[(0, i)].set_facecolor('#ED7D31')
    table_rf[(0, i)].set_text_props(weight='bold', color='white')
axes[1].set_title('Random Forest Performance Metrics',
                  fontweight='bold', fontsize=12, pad=20)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('comparison_tables.png', dpi=300, bbox_inches='tight')
plt.show()

# --- 图5：训练时间对比柱状图 ---
fig, ax = plt.subplots(figsize=(12, 6))

x_dt = np.arange(len(df_dt))
x_rf = np.arange(len(df_rf))
width = 0.35

bars1 = ax.bar(x_dt - width / 2, df_dt['train_time'], width,
               label='Decision Tree', color='skyblue', edgecolor='black', alpha=0.8)
bars2 = ax.bar(x_rf + width / 2, df_rf['train_time'], width,
               label='Random Forest', color='lightcoral', edgecolor='black', alpha=0.8)

ax.set_xlabel('Hyperparameter Configuration', fontsize=12)
ax.set_ylabel('Training Time (seconds)', fontsize=12)
ax.set_title('Training Time Comparison Across Hyperparameters', fontsize=14, fontweight='bold')
ax.set_xticks(np.arange(max(len(df_dt), len(df_rf))))
ax.set_xticklabels([f"DT: {d}\nRF: {n}" for d, n in
                    zip(df_dt['max_depth'], df_rf['n_estimators'])],
                   rotation=45, ha='right', fontsize=9)
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('training_time_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================
# 4. 统计摘要
# ============================================
print("\n" + "=" * 50)
print("Summary Statistics")
print("=" * 50)

print("\nDecision Tree:")
print(df_dt.describe())

print("\nRandom Forest:")
print(df_rf.describe())

# 保存数据到 CSV
df_dt.to_csv('dt_hyperparameter_comparison.csv', index=False)
df_rf.to_csv('rf_hyperparameter_comparison.csv', index=False)
print("\n✅ Results saved to CSV files!")

#---------------------------

# 4. Model Evaluation
# 4.1 classification performance
accuracy_dict = {}
for name, model in models.items():
    y_pred = model.predict(X_test_enc)
    acc = accuracy_score(y_test, y_pred)
    accuracy_dict[name] = acc

    print(f"\n=== {name} ===")
    print("Accuracy:", acc)
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred, target_names=["no", "yes"]))


accuracy_dict_no_dur = {}
for name, model in models_no_dur.items():
    y_pred = model.predict(X_test_no_dur_enc)
    acc = accuracy_score(y_test, y_pred)
    accuracy_dict_no_dur[name] = acc

    print(f"\n=== {name} ===")
    print("Accuracy:", acc)
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred, target_names=["no", "yes"]))

# 4.2 Confusion Matrix
for name, model in models.items():
    y_pred = model.predict(X_test_enc)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=["no", "yes"], yticklabels=["no", "yes"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"{name} Confusion Matrix")
    plt.show()

# 4.3 learning curve
def plot_learning_curve(model, X, y, title="Learning Curve", cv=5, scoring='accuracy',
                        train_sizes=np.linspace(0.1, 1.0, 5), ax=None):
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, cv=cv, scoring=scoring, train_sizes=train_sizes, n_jobs=-1
    )
    # Calculate Mean and Variance
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    val_scores_mean = np.mean(val_scores, axis=1)
    val_scores_std = np.std(val_scores, axis=1)

    # print X and Y value of each point
    print("\n=== point value ===")
    for i, size in enumerate(train_sizes):
        print(f"Train size: {size}, "
              f"Train score: {train_scores_mean[i]:.4f}, "
              f"Validation score: {val_scores_mean[i]:.4f}")

    # Plot on the Specified Subplot
    if ax is None:
        ax = plt.gca()

    ax.fill_between(train_sizes, train_scores_mean - train_scores_std,
                    train_scores_mean + train_scores_std, alpha=0.1, color="r")
    ax.fill_between(train_sizes, val_scores_mean - val_scores_std,
                    val_scores_mean + val_scores_std, alpha=0.1, color="g")
    ax.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    ax.plot(train_sizes, val_scores_mean, 'o-', color="g", label="Validation score")
    ax.set_title(title)
    ax.set_xlabel("Training examples")
    ax.set_ylabel(scoring)
    ax.legend(loc="best")
    ax.grid(True)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Decision Tree learning curve
best_dt_temp = DecisionTreeClassifier(**best_dt_params, random_state=42)
plot_learning_curve(best_dt_temp, X_train_val_enc, y_train_val, title="Decision Tree Learning Curve", ax=axes[0])

# Random Forest learning curve
best_rf_temp = RandomForestClassifier(**best_rf_params, random_state=42, n_jobs=-1)
plot_learning_curve(best_rf_temp, X_train_val_enc, y_train_val, title="Random Forest Learning Curve", ax=axes[1])

plt.tight_layout()
plt.show()

# 4.4
# 计算 Decision Tree 的交叉验证 F1-score
dt_cv_scores = cross_val_score(
    DecisionTreeClassifier(**best_dt_params, random_state=42),
    X_train_val_enc, y_train_val,
    cv=5, scoring='f1', n_jobs=-1
)
print("Decision Tree 5-Fold CV F1 Score: Mean = {:.3f}, Std = {:.3f}".format(
    dt_cv_scores.mean(), dt_cv_scores.std())
)

# 计算 Random Forest 的交叉验证 F1-score
rf_cv_scores = cross_val_score(
    RandomForestClassifier(**best_rf_params, random_state=42, n_jobs=-1),
    X_train_val_enc, y_train_val,
    cv=5, scoring='f1', n_jobs=-1
)
print("Random Forest 5-Fold CV F1 Score: Mean = {:.3f}, Std = {:.3f}".format(
    rf_cv_scores.mean(), rf_cv_scores.std())
)

plt.boxplot([dt_cv_scores, rf_cv_scores], tick_labels=['Decision Tree', 'Random Forest'])
plt.ylabel("F1 Score")
plt.title("Cross-Validation F1 Score Distribution")
plt.savefig("cv_f1_scores.png", dpi=300)

# 4.5 Feature Importance Analysis

# Get the Feature Column Names After One-Hot Encoding
onehot_columns = preprocessor.named_transformers_["onehot"].get_feature_names_out(categorical_features)
numeric_features = [col for col in X_train.columns if col not in categorical_features]
feature_names = list(onehot_columns) + numeric_features

# Decision Tree important feature
dt_importance = pd.Series(best_dt.feature_importances_, index=feature_names).sort_values(ascending=False)

# Random Forest important feature
rf_importance = pd.Series(best_rf.feature_importances_, index=feature_names).sort_values(ascending=False)

# Select the Top 20 Important Features of the Random Forest
top_n = 20
rf_top = rf_importance.head(top_n)

# Realign Decision Tree Features According to Random Forest Order
dt_values = dt_importance.reindex(rf_top.index).fillna(0)
rf_values = rf_top  # already in the top_n order.

y = np.arange(len(rf_top))
height = 0.4
plt.figure(figsize=(12,8))
plt.barh(y - height/2, dt_values, height=height, alpha=0.7, label='Decision Tree', color='skyblue')
plt.barh(y + height/2, rf_values, height=height, alpha=0.7, label='Random Forest', color='salmon')
plt.yticks(y, rf_top.index)
plt.gca().invert_yaxis()
plt.xlabel("Feature Importance")
plt.title("Feature Importance Comparison (Ordered by Random Forest) With Duration")
plt.legend()
plt.tight_layout()
plt.show()

#---no dur
# Get the Feature Column Names After One-Hot Encoding
onehot_columns = preprocessor_no_dur.named_transformers_["onehot"].get_feature_names_out(categorical_features)
numeric_features = [col for col in X_train_no_dur.columns if col not in categorical_features]
feature_names = list(onehot_columns) + numeric_features

# Decision Tree important feature
dt_importance = pd.Series(best_dt_no_dur.feature_importances_, index=feature_names).sort_values(ascending=False)

# Random Forest important feature
rf_importance = pd.Series(best_rf_no_dur.feature_importances_, index=feature_names).sort_values(ascending=False)

# Select the Top 20 Important Features of the Random Forest
top_n = 20
rf_top = rf_importance.head(top_n)

# Realign Decision Tree Features According to Random Forest Order
dt_values = dt_importance.reindex(rf_top.index).fillna(0)
rf_values = rf_top  # already in the top_n order.

y = np.arange(len(rf_top))
height = 0.4
plt.figure(figsize=(12,8))
plt.barh(y - height/2, dt_values, height=height, alpha=0.7, label='Decision Tree', color='skyblue')
plt.barh(y + height/2, rf_values, height=height, alpha=0.7, label='Random Forest', color='salmon')
plt.yticks(y, rf_top.index)
plt.gca().invert_yaxis()
plt.xlabel("Feature Importance")
plt.title("Feature Importance Comparison (Ordered by Random Forest) without Duration")
plt.legend()
plt.tight_layout()
plt.show()


# 5. Model Complexity and Efficiency Analysis
def model_size_in_mb(model, filename="temp_model.joblib"):
    dump(model, filename)
    size_mb = os.path.getsize(filename) / (1024 * 1024)
    os.remove(filename)
    return size_mb

complexity_data = []
for name, model in models.items():
    # Count total nodes (for tree-based models)
    if name == "Decision Tree":
        n_nodes = model.tree_.node_count
        max_depth = model.tree_.max_depth
    elif name == "Random Forest":
        n_nodes = sum(est.tree_.node_count for est in model.estimators_)
        max_depth = np.mean([est.tree_.max_depth for est in model.estimators_])

    # Measure inference time on test set
    start = time.time()
    _ = model.predict(X_test_enc)
    infer_time = time.time() - start

    # Measure model file size
    size_mb = model_size_in_mb(model)

    complexity_data.append({
        "Model": name,
        "Nodes": n_nodes,
        "Avg Depth": round(max_depth, 2),
        "Model Size (MB)": round(size_mb, 3),
        "Inference Time (s)": round(infer_time, 3)
    })

complexity_df = pd.DataFrame(complexity_data)
print("\n=== Model Complexity and Efficiency ===")
print(complexity_df.to_string(index=False))

# Visualization: model size vs accuracy
plt.figure(figsize=(6, 4))
plt.scatter(complexity_df["Model Size (MB)"], [accuracy_dict[m] for m in complexity_df["Model"]], s=150)
for i, row in complexity_df.iterrows():
    plt.text(row["Model Size (MB)"] + 0.1, accuracy_dict[row["Model"]], row["Model"])
plt.xlabel("Model Size (MB)")
plt.ylabel("Accuracy")
plt.title("Model Size vs Accuracy Trade-off")
plt.grid(True)
plt.show()

# 6. Hyperparameter Sensitivity Visualization

dt_depths = [5, 10, 15, 20, 25, 30, None]
dt_f1_scores = []
for d in dt_depths:
    model = DecisionTreeClassifier(max_depth=d, random_state=42)
    model.fit(X_train_enc, y_train)
    pred = model.predict(X_val_enc)
    dt_f1_scores.append(f1_score(y_val, pred))

# --- Random Forest: n_estimators vs F1 ---
rf_n_estimators = [10, 50, 100, 200, 250, 300]
rf_f1_scores = []
for n in rf_n_estimators:
    model = RandomForestClassifier(n_estimators=n, random_state=42, n_jobs=-1)
    model.fit(X_train_enc, y_train)
    pred = model.predict(X_val_enc)
    rf_f1_scores.append(f1_score(y_val, pred))

# --- Combine plots ---
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Left: Decision Tree
axes[0].plot([str(d) for d in dt_depths], dt_f1_scores, marker='o', label="Decision Tree", color="blue")
axes[0].set_xlabel("max_depth")
axes[0].set_ylabel("Validation F1 Score")
axes[0].set_title("Decision Tree Hyperparameter Sensitivity")
axes[0].grid(True)
axes[0].legend()

# Right: Random Forest
axes[1].plot(rf_n_estimators, rf_f1_scores, marker='o', color='orange', label="Random Forest")
axes[1].set_xlabel("n_estimators")
axes[1].set_ylabel("Validation F1 Score")
axes[1].set_title("Random Forest Hyperparameter Sensitivity")
axes[1].grid(True)
axes[1].legend()

plt.tight_layout()
plt.show()

# 7. ROC Curve and AUC Comparison
from sklearn.metrics import roc_curve, roc_auc_score

plt.figure(figsize=(6,5))
for name, model in models.items():
    y_prob = model.predict_proba(X_test_enc)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")

plt.plot([0,1], [0,1], 'k--', label="Random Guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.grid(True)
plt.show()

#----no dur
plt.figure(figsize=(6,5))
for name, model in models_no_dur.items():
    y_prob = model.predict_proba(X_test_no_dur_enc)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")

plt.plot([0,1], [0,1], 'k--', label="Random Guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison Without Duration")
plt.legend()
plt.grid(True)
plt.show()

# ROC and Precision–Recall Curves
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

# 模型预测概率
y_prob_dt = best_dt.predict_proba(X_test_enc)[:, 1]
y_prob_rf = best_rf.predict_proba(X_test_enc)[:, 1]

# --- ROC ---
fpr_dt, tpr_dt, _ = roc_curve(y_test, y_prob_dt)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
auc_dt = auc(fpr_dt, tpr_dt)
auc_rf = auc(fpr_rf, tpr_rf)

# --- Precision-Recall ---
prec_dt, rec_dt, _ = precision_recall_curve(y_test, y_prob_dt)
prec_rf, rec_rf, _ = precision_recall_curve(y_test, y_prob_rf)
ap_dt = average_precision_score(y_test, y_prob_dt)
ap_rf = average_precision_score(y_test, y_prob_rf)

# --- 绘图 ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 左图：ROC
axes[0].plot(fpr_dt, tpr_dt, label=f"Decision Tree (AUC = {auc_dt:.3f})", lw=2)
axes[0].plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC = {auc_rf:.3f})", lw=2)
axes[0].plot([0, 1], [0, 1], 'k--', lw=1)
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate")
axes[0].set_title("ROC Curve Comparison")
axes[0].legend()
axes[0].grid(True)

# 右图：Precision–Recall
axes[1].plot(rec_dt, prec_dt, label=f"Decision Tree (AP = {ap_dt:.3f})", lw=2)
axes[1].plot(rec_rf, prec_rf, label=f"Random Forest (AP = {ap_rf:.3f})", lw=2)
axes[1].set_xlabel("Recall")
axes[1].set_ylabel("Precision")
axes[1].set_title("Precision–Recall Curve Comparison")
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()

# Model Calibration and Reliability
# 获取预测概率
y_prob_dt = best_dt.predict_proba(X_test_enc)[:, 1]  # Decision Tree
y_prob_rf = best_rf.predict_proba(X_test_enc)[:, 1]  # Random Forest

# 计算校准曲线
prob_true_dt, prob_pred_dt = calibration_curve(y_test, y_prob_dt, n_bins=10)
prob_true_rf, prob_pred_rf = calibration_curve(y_test, y_prob_rf, n_bins=10)

# 绘图
plt.figure(figsize=(6,5))
plt.plot(prob_pred_dt, prob_true_dt, marker='o', label=f"Decision Tree (Brier={brier_score_loss(y_test, y_prob_dt):.3f})")
plt.plot(prob_pred_rf, prob_true_rf, marker='s', label=f"Random Forest (Brier={brier_score_loss(y_test, y_prob_rf):.3f})")
plt.plot([0,1], [0,1], 'k--', label="Perfect Calibration")
plt.xlabel("Mean Predicted Probability")
plt.ylabel("Fraction of Positives")
plt.title("Model Calibration Curve")
plt.legend()
plt.grid(True)
plt.show()