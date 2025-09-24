import joblib
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

model_path = "rf_no2_downscaler.pkl"
rf = joblib.load(model_path)
print("Random Forest model loaded\n")

print("Random Forest Model Summary:")
print(f"Number of trees (n_estimators): {len(rf.estimators_)}")
print(f"Number of features: {rf.n_features_in_}")
print(f"Feature names: ['latitude', 'longitude']")
print(f"Max depth: {rf.max_depth}")
try:
    print(f"Number of samples used to train: {rf.n_samples_}")
except AttributeError:
    print("Number of samples used: Not stored in sklearn RandomForestRegressor")
print(f"Random state: {rf.random_state}\n")

if 'X_val' in globals() and 'y_val' in globals():
    y_pred = rf.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    r2 = r2_score(y_val, y_pred)
    print(f"Validation RMSE: {rmse:.6f}")
    print(f"Validation R²: {r2:.4f}\n")

estimator = rf.estimators_[0]

plt.figure(figsize=(15, 8))
tree.plot_tree(
    estimator,
    feature_names=['latitude', 'longitude'],
    filled=True,
    max_depth=3,
    fontsize=10
)
plt.title("Random Forest: First Tree (Top 3 Levels)")
plt.savefig("rf_first_tree_top3.png", dpi=300)
plt.show()

importance = rf.feature_importances_
features = ['latitude', 'longitude']

plt.figure(figsize=(6, 4))
plt.bar(features, importance, color=['skyblue', 'orange'])
plt.title("Feature Importance in Random Forest")
plt.ylabel("Importance")
for i, v in enumerate(importance):
    plt.text(i, v + 0.01, f"{v:.3f}", ha='center', fontsize=10)
plt.savefig("rf_feature_importance.png", dpi=300)
plt.show()
