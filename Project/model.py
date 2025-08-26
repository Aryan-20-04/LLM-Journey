import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import VotingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, classification_report

#=====Load Data=========
df=pd.read_csv('heart.csv')
X=df.drop("HeartDisease", axis=1)
y=df["HeartDisease"]
#======================

#======Categorical========
categorical_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
X_encoded = X.copy()
for col in categorical_cols:
    le = LabelEncoder()
    X_encoded[col] = le.fit_transform(X_encoded[col])
#=======================

#====Model Training======
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42, stratify=y
)
# Random Forest (tuned)
rfc = RandomForestClassifier(
    n_estimators=107,
    max_depth=20,
    min_samples_split=2,
    min_samples_leaf=1,
    criterion='entropy',
    random_state=42
)

# XGBoost (example tuned params)
xgb = XGBClassifier(
    n_estimators=500,
    max_depth=4,
    learning_rate=0.05,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)

# CatBoost (example tuned params)
cbc = CatBoostClassifier(
    iterations=500,
    depth=4,
    learning_rate=0.05,
    verbose=0,
    random_seed=42
)
#type: ignore
ensemble = VotingClassifier(
    estimators=[('rfc', rfc), ('xgb', xgb), ('cbc', cbc)],
    voting='soft'  # 'soft' uses predicted probabilities
)
ensemble.fit(X_train,y_train)

#========================

#====Testing=====
y_pred = ensemble.predict(X_test)

# Accuracy & Classification
print("Ensemble Test Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

#=========================