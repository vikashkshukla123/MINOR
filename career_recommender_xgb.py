

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import warnings

warnings.filterwarnings("ignore")

# ---------- 1) LOAD DATASET ----------
df = pd.read_csv("career_data_large.csv")  # your dataset

# ---------- 2) AUTOMATIC CATEGORICAL ENCODING ----------
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
if 'Recommended_Career' not in categorical_cols:
    categorical_cols.append('Recommended_Career')

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# ---------- 3) FEATURES AND TARGET ----------
X = df.drop(['Name', 'Recommended_Career'], axis=1, errors='ignore')
y = df['Recommended_Career']

# ---------- 4) TRAIN MODEL ----------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = XGBClassifier(
    n_estimators=400,
    learning_rate=0.03,
    max_depth=8,
    subsample=0.85,
    colsample_bytree=0.85,
    random_state=42,
    use_label_encoder=False,
    eval_metric='mlogloss'
)
model.fit(X_train, y_train)

# ---------- 5) USER INPUT FUNCTION ----------
def predict_from_input():
    # Numeric inputs
    age = int(input("Age: "))
    cgpa = float(input("College CGPA (0–10): "))
    programming = int(input("Programming (1–10): "))
    data_analysis = int(input("Data Analysis (1–10): "))
    ai_ml = int(input("AI/ML (1–10): "))
    math = int(input("Math (1–10): "))
    science = int(input("Science (1–10): "))
    creativity = int(input("Creativity (1–10): "))
    communication = int(input("Communication (1–10): "))
    teamwork = int(input("Teamwork (1–10): "))
    problem_solving = int(input("Problem Solving (1–10): "))

    # Categorical inputs
    gender = input("Gender (Male/Female/Other): ")
    edu_stream = input("Education Stream (e.g., Computer Science): ")
    degree_level = input("Degree Level (Diploma/Bachelors/Masters/PhD): ")
    interest = input("Interest Area (e.g., AI, Software, Design, etc.): ")
    personality = input("Personality Type (Introvert/Extrovert/Ambivert): ")

    # Encode categorical safely
    def safe_encode(le, value):
        if value not in le.classes_:
            return 0
        return le.transform([value])[0]

    input_data = pd.DataFrame([{
        'Age': age,
        'Gender': safe_encode(label_encoders['Gender'], gender),
        'CGPA': cgpa,
        'Programming': programming,
        'Data_Analysis': data_analysis,
        'AI_ML': ai_ml,
        'Math': math,
        'Science': science,
        'Creativity': creativity,
        'Communication': communication,
        'Teamwork': teamwork,
        'Problem_Solving': problem_solving,
        'Education_Stream': safe_encode(label_encoders['Education_Stream'], edu_stream),
        'Degree_Level': safe_encode(label_encoders['Degree_Level'], degree_level),
        'Interest_Area': safe_encode(label_encoders['Interest_Area'], interest),
        'Personality_Type': safe_encode(label_encoders['Personality_Type'], personality)
    }], columns=X.columns)

    # Predict probabilities for all possible careers
    probs = model.predict_proba(input_data)[0]

    # Get top 3 careers
    top3_idx = np.argsort(probs)[-3:][::-1]
    top3_conf = probs[top3_idx]
    top3_labels = label_encoders['Recommended_Career'].inverse_transform(top3_idx)

    # Show top 3 results neatly
    print("\n🎓 Top 3 Career Recommendations:")
    for i in range(3):
        print(f"{i+1}) {top3_labels[i]} — Confidence: {top3_conf[i]*100:.2f}%")

# ---------- 6) RUN ----------
if __name__ == "__main__":
    predict_from_input()  