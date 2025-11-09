# # ==========================================
# # Generate Synthetic Career Dataset (5000 rows)
# # ==========================================
# import pandas as pd
# import numpy as np
# import random

# # Set random seed for reproducibility
# random.seed(42)
# np.random.seed(42)

# # Possible categories
# genders = ["Male", "Female"]
# streams = [
#     "Computer Science", "Information Technology", "Electrical Engineering",
#     "Mechanical Engineering", "Civil Engineering", "Data Science",
#     "Business Administration", "Commerce", "Graphic Design", "Electronics"
# ]
# degrees = ["Bachelors", "Masters"]
# interest_areas = ["AI", "Software", "Design", "Research", "Management"]
# personalities = ["Introvert", "Extrovert", "Ambivert"]

# # Possible careers mapping (simplified for realism)
# career_map = {
#     "AI": ["Data Scientist", "AI Engineer", "Machine Learning Engineer", "AI Researcher"],
#     "Software": ["Software Engineer", "Full Stack Developer", "Software Developer"],
#     "Design": ["UI/UX Designer", "Graphic Designer", "Automobile Engineer"],
#     "Research": ["Data Engineer", "Research Scientist", "Electronics Engineer"],
#     "Management": ["HR Manager", "Project Manager", "Financial Analyst"]
# }

# # Generate random synthetic dataset
# rows = []
# for i in range(5000):
#     name = f"Person_{i+1}"
#     age = random.randint(21, 30)
#     gender = random.choice(genders)
#     stream = random.choice(streams)
#     degree = random.choice(degrees)
#     cgpa = round(np.random.uniform(6.5, 9.9), 2)

#     # Skills (1–10)
#     programming = random.randint(1, 10)
#     data_analysis = random.randint(1, 10)
#     ai_ml = random.randint(1, 10)
#     math = random.randint(1, 10)
#     science = random.randint(1, 10)
#     creativity = random.randint(1, 10)
#     communication = random.randint(1, 10)
#     teamwork = random.randint(1, 10)
#     problem_solving = random.randint(1, 10)

#     interest = random.choice(interest_areas)
#     personality = random.choice(personalities)

#     # Recommended career based on strongest interest
#     career = random.choice(career_map[interest])

#     rows.append([
#         name, age, gender, stream, degree, cgpa,
#         programming, data_analysis, ai_ml, math, science,
#         creativity, communication, teamwork, problem_solving,
#         interest, personality, career
#     ])

# # Create DataFrame
# columns = [
#     "Name", "Age", "Gender", "Education_Stream", "Degree_Level", "CGPA",
#     "Programming", "Data_Analysis", "AI_ML", "Math", "Science",
#     "Creativity", "Communication", "Teamwork", "Problem_Solving",
#     "Interest_Area", "Personality_Type", "Recommended_Career"
# ]

# df = pd.DataFrame(rows, columns=columns)

# # Save to CSV
# df.to_csv("career_data_large.csv", index=False, encoding="utf-8")
# print("✅ Synthetic dataset 'career_data_large.csv' created successfully with", len(df), "rows.")
# print(df.head())
# ==============================================
# SMART CAREER DATASET GENERATOR v3
# Large-scale, realistic dataset for career prediction
# ==============================================

import pandas as pd
import numpy as np
import random

# For reproducibility
random.seed(42)
np.random.seed(42)

# Number of samples to generate (you can increase to 100000+)
NUM_SAMPLES = 50000

# ---------- CATEGORY OPTIONS ----------
genders = ["Male", "Female", "Other"]

streams = [
    "Computer Science", "Information Technology", "Electrical Engineering",
    "Mechanical Engineering", "Civil Engineering", "Data Science",
    "Business Administration", "Commerce", "Graphic Design",
    "Electronics", "Physics", "Mathematics", "Architecture", "Economics"
]

degrees = ["Diploma", "Bachelors", "Masters", "PhD"]

interest_areas = ["AI", "Software", "Design", "Research", "Management", "Engineering"]
personalities = ["Introvert", "Extrovert", "Ambivert"]

career_map = {
    "AI": ["AI Engineer", "Data Scientist", "ML Engineer", "AI Researcher"],
    "Software": ["Software Engineer", "Full Stack Developer", "Backend Developer"],
    "Design": ["UI/UX Designer", "Graphic Designer", "Product Designer"],
    "Research": ["Research Scientist", "Data Engineer", "Professor"],
    "Management": ["HR Manager", "Project Manager", "Financial Analyst"],
    "Engineering": ["Electrical Engineer", "Mechanical Engineer", "Civil Engineer"]
}

# ---------- CAREER ASSIGNMENT LOGIC ----------
def assign_career(row):
    interest = row["Interest_Area"]
    stream = row["Education_Stream"]
    degree = row["Degree_Level"]
    cgpa = row["CGPA"]
    exp = row["Years_of_Experience"]
    projects = row["Projects_Done"]
    prog = row["Programming"]
    data = row["Data_Analysis"]
    ai_ml = row["AI_ML"]
    creativity = row["Creativity"]
    comm = row["Communication"]
    teamwork = row["Teamwork"]

    # AI / Data roles
    if interest == "AI" or "Data" in stream:
        if ai_ml > 7 or data > 7:
            if exp >= 2 or projects >= 4:
                return random.choice(["Data Scientist", "AI Engineer"])
            else:
                return "AI Researcher"

    # Software roles
    if interest == "Software" or "Computer" in stream or "IT" in stream:
        if prog >= 7:
            if projects >= 3:
                return "Full Stack Developer"
            else:
                return "Software Engineer"
        else:
            return "Backend Developer"

    # Design roles
    if interest == "Design" or "Design" in stream:
        if creativity >= 7:
            return random.choice(["UI/UX Designer", "Graphic Designer"])
        else:
            return "Product Designer"

    # Management roles
    if interest == "Management" or "Business" in stream or "Commerce" in stream:
        if comm + teamwork >= 13:
            if exp > 2:
                return "Project Manager"
            else:
                return "HR Manager"
        else:
            return "Financial Analyst"

    # Engineering roles
    if interest == "Engineering" or "Electrical" in stream or "Mechanical" in stream:
        if exp > 3:
            return random.choice(["Mechanical Engineer", "Electrical Engineer"])
        else:
            return "Civil Engineer"

    # Research/Academia
    if interest == "Research" or degree in ["Masters", "PhD"]:
        if cgpa > 8.0:
            return "Research Scientist"
        else:
            return "Professor"

    return random.choice(sum(career_map.values(), []))


# ---------- DATA GENERATION ----------
rows = []
for i in range(NUM_SAMPLES):
    name = f"Person_{i+1}"
    age = random.randint(20, 40)
    gender = random.choice(genders)
    stream = random.choice(streams)
    degree = random.choice(degrees)
    cgpa = round(np.random.uniform(5.5, 10.0), 2)
    experience = random.randint(0, 10)

    # Projects based on degree + experience
    if degree == "Diploma":
        projects = random.randint(1, 4)
    elif degree == "Bachelors":
        projects = random.randint(2, 6)
    elif degree == "Masters":
        projects = random.randint(3, 8)
    else:
        projects = random.randint(4, 10)

    # Skills (biased by stream)
    base_skill = random.randint(4, 7)
    if "Computer" in stream or "IT" in stream or "Data" in stream:
        programming = random.randint(6, 10)
        data_analysis = random.randint(5, 10)
        ai_ml = random.randint(4, 10)
    else:
        programming = random.randint(1, 7)
        data_analysis = random.randint(2, 8)
        ai_ml = random.randint(1, 7)

    math = random.randint(4, 10)
    science = random.randint(4, 10)
    creativity = random.randint(1, 10)
    communication = random.randint(1, 10)
    teamwork = random.randint(1, 10)
    problem_solving = random.randint(1, 10)

    interest = random.choice(interest_areas)
    personality = random.choice(personalities)

    rows.append([
        name, age, gender, stream, degree, cgpa, experience, projects,
        programming, data_analysis, ai_ml, math, science, creativity,
        communication, teamwork, problem_solving, interest, personality
    ])

columns = [
    "Name", "Age", "Gender", "Education_Stream", "Degree_Level", "CGPA",
    "Years_of_Experience", "Projects_Done", "Programming", "Data_Analysis",
    "AI_ML", "Math", "Science", "Creativity", "Communication",
    "Teamwork", "Problem_Solving", "Interest_Area", "Personality_Type"
]

df = pd.DataFrame(rows, columns=columns)

# Assign recommended careers
df["Recommended_Career"] = df.apply(assign_career, axis=1)

# ---------- SAVE ----------
df.to_csv("career_data_large.csv", index=False, encoding="utf-8")

print(f"✅ Successfully generated {len(df):,} career records in 'career_data_large.csv'")
print("📊 Sample preview:")
print(df.head(10))
