import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
import shap

from joblib import dump
from sklearn.model_selection import train_test_split,learning_curve
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
from scipy.sparse import vstack


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

# 1.3 One-Hot encoding for categorical features
categorical_features = X.select_dtypes(include=["object"]).columns
preprocessor = ColumnTransformer(
    transformers=[
        ("onehot", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ],
    remainder="passthrough"
)
preprocessor.fit(X_train) # fit in training sets only
X_train_enc = preprocessor.transform(X_train)
X_val_enc   = preprocessor.transform(X_val)
X_test_enc  = preprocessor.transform(X_test)

# 2. Model Training and Hyperparameter Tuning
# ---- Decision Tree Hyperparameter Tuning ----
dt_params = {"max_depth": [5, 10, 15, None],
             "min_samples_split": [2, 5, 10, 15, 20]}
best_dt_score, best_dt_params = 0, None
for depth in dt_params["max_depth"]:
    for min_split in dt_params["min_samples_split"]:
        dt = DecisionTreeClassifier(max_depth=depth,
                                    min_samples_split=min_split,
                                    random_state=42)
        dt.fit(X_train_enc, y_train)
        y_val_pred = dt.predict(X_val_enc)
        f1 = f1_score(y_val, y_val_pred)
        if f1 > best_dt_score:
            best_dt_score = f1
            best_dt_params = {"max_depth": depth, "min_samples_split": min_split}
print("Decision Tree Best Parameters:", best_dt_params)

# ---- Random Forest Hyperparameter Tuning ----
rf_params = {"n_estimators": [50, 100, 200, 250],
             "max_depth": [5, 10, 15, None]}
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
        if f1 > best_rf_score:
            best_rf_score = f1
            best_rf_params = {"n_estimators": n, "max_depth": depth}
print("Random Forest Best Parameters:", best_rf_params)

# 3. Retraining with Best Parameters
X_train_val_enc = vstack([X_train_enc, X_val_enc])
y_train_val = np.concatenate([y_train, y_val])

best_dt = DecisionTreeClassifier(**best_dt_params, random_state=42)
best_dt.fit(X_train_val_enc, y_train_val)

best_rf = RandomForestClassifier(**best_rf_params, random_state=42, n_jobs=-1)
best_rf.fit(X_train_val_enc, y_train_val)

models = {"Decision Tree": best_dt, "Random Forest": best_rf}

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

# 4.4 Feature Importance Analysis

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
plt.title("Feature Importance Comparison (Ordered by Random Forest)")
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

# Decision Tree: max_depth vs F1
dt_depths = [5, 10, 15, None]
dt_f1_scores = []
for d in dt_depths:
    model = DecisionTreeClassifier(max_depth=d, random_state=42)
    model.fit(X_train_enc, y_train)
    pred = model.predict(X_val_enc)
    dt_f1_scores.append(f1_score(y_val, pred))

plt.figure(figsize=(6,4))
depth_labels = [str(d) for d in dt_depths]
plt.plot(depth_labels, dt_f1_scores, marker='o', label="Decision Tree")
plt.xlabel("max_depth")
plt.ylabel("Validation F1 Score")
plt.title("Decision Tree Hyperparameter Sensitivity")
plt.grid(True)
plt.legend()
plt.show()

# Random Forest: n_estimators vs F1
rf_n_estimators = [50, 100, 200, 250]
rf_f1_scores = []
for n in rf_n_estimators:
    model = RandomForestClassifier(n_estimators=n, random_state=42, n_jobs=-1)
    model.fit(X_train_enc, y_train)
    pred = model.predict(X_val_enc)
    rf_f1_scores.append(f1_score(y_val, pred))

plt.figure(figsize=(6,4))
plt.plot(rf_n_estimators, rf_f1_scores, marker='o', color='orange', label="Random Forest")
plt.xlabel("n_estimators")
plt.ylabel("Validation F1 Score")
plt.title("Random Forest Hyperparameter Sensitivity")
plt.grid(True)
plt.legend()
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

# 8. Model Interpretability with SHAP (robust version)
import shap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.sparse

# Prepare sample (200 rows)
if scipy.sparse.issparse(X_test_enc):
    X_sample = X_test_enc[:200].toarray()
else:
    X_sample = X_test_enc[:200]

# Convert to DataFrame for SHAP plotting
X_sample_df = pd.DataFrame(X_sample, columns=feature_names)

# Initialize TreeExplainer
explainer = shap.TreeExplainer(best_rf, feature_perturbation="tree_path_dependent")
shap_values = explainer.shap_values(X_sample_df)

# Determine correct SHAP values for plotting
if isinstance(shap_values, list):  # list means multi-class or sklearn binary
    if len(shap_values) == 2 and shap_values[0].shape == shap_values[1].shape:  # sklearn binary
        shap_plot_values = shap_values[1]  # positive class
    else:  # multi-class, choose class_idx=1 by default
        shap_plot_values = shap_values[1]
else:
    shap_plot_values = shap_values  # already 2D

# Verify shapes
print("SHAP values shape:", shap_plot_values.shape)
print("Data shape:", X_sample_df.shape)

# Plot SHAP summary
shap.summary_plot(shap_plot_values, X_sample_df, feature_names=feature_names, show=False)
plt.savefig("figures/shap_summary_rf.png", bbox_inches="tight", dpi=300)
plt.close()
