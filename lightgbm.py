import lightgbm as lgb
from sklearn.metrics import accuracy_score

lgbm_model = lgb.LGBMClassifier(
    boosting_type='gbdt',
    num_leaves=31,
    learning_rate=0.05,
    n_estimators=100,
    random_state=42
)

lgbm_model.fit(X_train, y_train)

lgbm_predictions = lgbm_model.predict(X_test)
print(f"LightGBM Accuracy: {accuracy_score(y_test, lgbm_predictions):.2%}")