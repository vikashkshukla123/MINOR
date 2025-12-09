# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.preprocessing import LabelEncoder
# from sklearn.metrics import classification_report
# import joblib
# import os

# # --- Configuration ---
# INPUT_FILE = "synthetic_20_job_binary_dataset.csv"
# MODEL_FILE = "career_recommender_model.pkl"
# TARGET_COLUMN = 'Occupation_Name'
# EXCLUDED_COLS = ['Occupation_Code', TARGET_COLUMN]

# def train_career_recommender():
#     """Trains the Random Forest model and saves it."""
    
#     # 1. Load Data
#     try:
#         df = pd.read_csv(INPUT_FILE)
#     except FileNotFoundError:
#         print(f"\n❌ ERROR: Input file '{INPUT_FILE}' not found.")
#         print("Please ensure your 'generate.py' script has run successfully.")
#         return None, None, None
    
#     # Check for required demographic columns (Age/CGPA)
#     required_new_cols = ['Age', 'CGPA']
#     if not all(col in df.columns for col in required_new_cols):
#         print(f"\n❌ CRITICAL ERROR: Missing 'Age' or 'CGPA' columns in the CSV.")
#         print("Please ensure you run the latest version of 'generate.py' which augments the data.")
#         return None, None, None

#     # 2. Prepare Features (X) and Target (y)
#     # X includes Age, CGPA, and all the binary attributes
#     X = df.drop(EXCLUDED_COLS, axis=1)
    
#     # Encode target labels (job names) into numerical format
#     le = LabelEncoder()
#     y = le.fit_transform(df[TARGET_COLUMN])
    
#     # 3. Split Data
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=42, stratify=y
#     )
    
#     # 4. Train Model (Random Forest Classifier)
#     print(f"Training Random Forest Classifier with {len(X.columns)} features and {len(le.classes_)} classes...")
#     model = RandomForestClassifier(
#         n_estimators=100, 
#         max_depth=20, 
#         random_state=42,
#         class_weight='balanced'
#     )
#     model.fit(X_train, y_train)
    
#     # 5. Evaluation
#     y_pred = model.predict(X_test)
#     print("\n--- Model Evaluation (Test Set) ---")
#     # Note: Full classification report is only possible if terminal is wide enough for 20 classes
#     print(classification_report(y_test, y_pred, target_names=le.classes_))
    
#     # 6. Save Model and Encoder
#     joblib.dump((model, le, list(X.columns)), MODEL_FILE)
#     print(f"✅ Training Complete. Model and Label Encoder saved to {MODEL_FILE}")
    
#     return model, le, list(X.columns)


# def get_user_input_profile(feature_cols):
#     """
#     Collects user input for Age, CGPA, and ALL binary attributes for full coverage.
#     """
    
#     print("\n--- User Profile Input ---")

#     # 1. Get Age and CGPA
#     while True:
#         try:
#             age = int(input("Enter your Age: "))
#             cgpa = float(input("Enter your CGPA (e.g., 7.5): "))
#             if 18 <= age <= 70 and 0 <= cgpa <= 10:
#                 break
#             else:
#                 print("Invalid range. Please enter realistic values.")
#         except ValueError:
#             print("Invalid input. Please enter numbers only.")

#     # Initialize the user profile vector
#     user_data = {}
    
#     # 2. Get Input for ALL Binary Attributes
#     # Exclude Age and CGPA since they were asked separately
#     binary_attributes = [col for col in feature_cols if col not in ['Age', 'CGPA']]
    
#     print(f"\n[Input your level (1=Present/High, 0=Absent/Low) for ALL {len(binary_attributes)} attributes.]")
#     print("This covers all Technology Skills, Knowledge, Work Styles, and Interests.")
    
#     for i, attr in enumerate(binary_attributes, 1):
#         # Format the attribute name for readability (e.g., 'SYSTEMS_ANALYSIS' -> 'Systems Analysis')
#         readable_attr = attr.replace('_', ' ').title()
        
#         while True:
#             try:
#                 score = int(input(f"({i}/{len(binary_attributes)}) Score for {readable_attr} (1/0): "))
#                 if score in [0, 1]:
#                     user_data[attr] = score
#                     break
#                 else:
#                     print("Please enter only 1 or 0.")
#             except ValueError:
#                 print("Invalid input. Please enter 1 or 0.")
                
#     # Add Age and CGPA back to the dictionary
#     user_data['Age'] = age
#     user_data['CGPA'] = cgpa
                
#     # Create a DataFrame from the user input (1 row)
#     user_df = pd.DataFrame([user_data])
    
#     # Ensure columns match the training order (CRITICAL STEP)
#     user_df = user_df[feature_cols]

#     return user_df


# def recommend_careers(model, label_encoder, user_df):
#     """Predicts the top 4 careers based on user input."""
    
#     # 1. Predict probabilities for all classes
#     probabilities = model.predict_proba(user_df)[0]
    
#     # 2. Get class indices and corresponding job names
#     job_indices = np.argsort(probabilities)[::-1] # Sort descending
#     job_names = label_encoder.inverse_transform(job_indices)
    
#     # 3. Create the recommendation list
#     recommendations = []
#     for rank in range(4):
#         job = job_names[rank]
#         prob = probabilities[job_indices[rank]]
#         recommendations.append((job, prob))
        
#     print("\n--- Top 4 Career Recommendations ---")
#     for i, (job, prob) in enumerate(recommendations, 1):
#         print(f"#{i}: {job} (Confidence Score: {prob:.2f})")
    
#     print("\nRecommendation Complete.")

# # --- Main Execution ---
# if __name__ == "__main__":
    
#     if os.path.exists(MODEL_FILE):
#         print(f"Loading trained model from {MODEL_FILE}...")
#         try:
#             model, le, feature_cols = joblib.load(MODEL_FILE)
#             print("Model loaded successfully.")
#         except Exception:
#             print("Error loading model. Retraining...")
#             model, le, feature_cols = train_career_recommender()
#     else:
#         model, le, feature_cols = train_career_recommender()

#     if model is not None:
#         user_profile_df = get_user_input_profile(feature_cols)
#         recommend_careers(model, le, user_profile_df)


# import pandas as pd
# import numpy as np
# import re
# from sklearn.neighbors import NearestNeighbors

# # --- STATIC MAPPINGS FOR ALL FEATURE CATEGORIES ---

# # 1. Differentiated/Skippable Binary Inputs (Technology Skills)
# TECH_BINARY_MAPS = {
#     'Analytical (Tech)': ['ADOBE', 'AHREFS_SITE_EXPLORER', 'AMAZON_REDSHIFT', 'BING_ADS', 'GOOGLE', 'GOOGLE_ADS', 'INFORMATICA', 'MICROSOFT_EXCEL', 'MICROSOFT_POWER_BI', 'MICROSOFT_SQL_SERVER_INTEGRATION_SERVICES_SSIS', 'MOZ_SEARCH_ENGINE_OPTIMIZATION_SEO', 'PANDAS', 'PYTORCH', 'R', 'SAS', 'SCIKIT_LEARN', 'SCREAMING_FROG_SEO_SPIDER', 'SEMRUSH', 'STRUCTURED_QUERY_SQL', 'TABLEAU', 'TENSORFLOW'],
#     'Creative (Tech)': ['ADOBE_AFTER_EFFECTS', 'ADOBE_CREATIVE_CLOUD', 'ADOBE_DREAMWEAVER', 'ADOBE_ILLUSTRATOR', 'ADOBE_INDESIGN', 'ADOBE_PHOTOSHOP', 'ADOBE_PREMIERE_PRO', 'AUTODESK_MAYA', 'CASCADING_STYLE_SHEETS_CSS', 'FIGMA', 'GOOGLE_ANGULAR', 'HYPERTEXT_MARKUP_LANGUAGE_HTML', 'JAVASCRIPT', 'MICROSOFT_DIRECTX', 'REACT', 'UNITY_TECHNOLOGIES_UNITY', 'UNREAL_TECHNOLOGY_UNREAL_ENGINE', 'VULKAN_GRAPHICS_API', 'WEB_APPLICATION', 'WORDPRESS'],
#     'Logical (Tech)': ['APACHE_HADOOP', 'APACHE_KAFKA', 'APACHE_SPARK', 'BASH', 'C', 'C#', 'C++', 'MICROSOFT_SQL_SERVER', 'MYSQL', 'NODE_JS', 'NOSQL', 'ORACLE_JAVA', 'ORACLE_PL_SQL', 'POSTGRESQL', 'PYTHON', 'RESTFUL_API', 'SCALA', 'TYPESCRIPT'],
#     'Practical (Tech)': ['AMAZON_WEB_SERVICES_AWS', 'ANSIBLE', 'CISCO', 'DOCKER', 'FIREWALL', 'GIT', 'IBM_TERRAFORM', 'JENKINS_CI', 'KUBERNETES', 'LINUX', 'MICROSOFT_ACTIVE_DIRECTORY', 'MICROSOFT_AZURE', 'MICROSOFT_POWERSHELL', 'MICROSOFT_SYSTEM_CENTER_CONFIGURATION_MANAGER', 'MICROSOFT_WINDOWS', 'MICROSOFT_WINDOWS_SERVER', 'SERVICENOW', 'OPERATING_SYSTEM_SOFTWARE', 'ORACLE_CLOUD', 'PHP', 'POSTMAN', 'RED_HAT_ENTERPRISE_LINUX', 'SALESFORCE', 'SAP', 'SELENIUM', 'UNIX'],
#     'Social (Tech)': ['ATLASSIAN_JIRA', 'GOOGLE', 'INFORMATION_TECHNICAL_PROJECT_MANAGER', 'MICROSOFT_OFFICE', 'MICROSOFT_OUTLOOK', 'MICROSOFT_POWERPOINT', 'MICROSOFT_SHAREPOINT', 'MICROSOFT_VISIO', 'SALES_ENGINEER', 'SCRUM_MASTER'],
# }

# # 2. Mandatory Binary Inputs (Comprehensive Skills, Knowledge, Work Styles, Interests, Education)
# MANDATORY_BINARY_MAPS = {
#     'Comprehensive Skills (Mandatory)': ['SPEAKING', 'READING_COMPREHENSION', 'ACTIVE_LISTENING', 'CRITICAL_THINKING', 'SYSTEMS_ANALYSIS', 'SYSTEMS_EVALUATION', 'ACTIVE_LEARNING', 'JUDGMENT_DECISION_MAKING', 'COMPLEX_PROBLEM_SOLVING', 'MONITORING', 'WRITING'],
#     'Knowledge (Mandatory)': ['COMPUTERS_ELECTRONICS', 'CUSTOMER_PERSONAL_SERVICE', 'ENGLISH_LANGUAGE', 'MATHEMATICS', 'ADMINISTRATION_MANAGEMENT', 'EDUCATION_TRAINING', 'ECONOMICS_ACCOUNTING', 'LAW_GOVERNMENT', 'SALES_MARKETING', 'HUMAN_RESOURCES', 'THERAPY_COUNSELING', 'PUBLIC_SAFETY_SECURITY', 'DESIGN', 'FINE_ARTS', 'COMMUNICATIONS_MEDIA'],
#     'Work Styles (Mandatory)': ['ANALYTICAL_THINKING', 'ATTENTION_TO_DETAIL', 'DEPENDABILITY', 'COOPERATION', 'INTEGRITY', 'INITIATIVE', 'PERSISTENCE', 'INNOVATION', 'STRESS_TOLERANCE', 'ADAPTABILITY_FLEXIBILITY', 'INDEPENDENCE', 'ACHIEVEMENT_EFFORT', 'CONCERN_FOR_OTHERS', 'SELF_CONTROL', 'LEADERSHIP'],
#     'Interests (Mandatory)': ['INVESTIGATIVE', 'CONVENTIONAL', 'ARTISTIC'],
#     'Education (Mandatory)': ['B_TECH', 'M_TECH'],
# }


# # --- UTILITY FUNCTIONS ---

# def clean_attribute_name(attribute_str):
#     # Standard cleaning logic to match features in the training data
#     clean_attribute = re.sub(r'(software|and|the|language|designers|for|occupation|making|training|personal|service)', '', attribute_str, flags=re.IGNORECASE).strip()
#     clean_attribute = re.sub(r'[\s\W]+', '_', clean_attribute).strip('_').upper()
#     return clean_attribute

# def safe_input(prompt, is_float=False, min_val=0, max_val=1):
#     """Handles input validation for binary or continuous data."""
#     while True:
#         try:
#             value = input(prompt)
#             if is_float:
#                 val = float(value)
#                 if min_val <= val <= max_val:
#                     return val
#                 else:
#                     print(f"Input must be between {min_val} and {max_val}.")
#             else:
#                 val = int(value)
#                 if val in [0, 1]:
#                     return val
#                 else:
#                     print("Invalid input. Please enter 1 or 0.")
#         except ValueError:
#             print("Invalid input. Please enter a valid number.")

# def select_categories(category_maps):
#     """Presents a menu for the user to select which categories they want to input."""
    
#     print("\n[ STEP 2A: SELECT IN-DEMAND TECHNOLOGY SKILL CATEGORIES TO ENTER ]")
#     print("Please select the categories you want to enter skills/attributes for. (e.g., 1, 3)")
    
#     category_names = list(category_maps.keys())
#     for i, name in enumerate(category_names):
#         print(f"  {i+1}. {name.replace(' (Tech)', '')}")
        
#     while True:
#         selection_input = input("\nEnter the numbers of the Tech categories you want to input (comma-separated): > ")
#         try:
#             # Parse input and convert to 0-indexed list
#             selected_indices = [int(x.strip()) - 1 for x in selection_input.split(',')]
            
#             # Validate indices
#             valid_indices = [i for i in selected_indices if 0 <= i < len(category_names)]
            
#             if not valid_indices:
#                 print("No valid categories selected. Please enter at least one or try again.")
#                 continue

#             selected_categories = [category_names[i] for i in valid_indices]
#             print(f"\n✅ You chose to input for: {', '.join([c.replace(' (Tech)', '') for c in selected_categories])}")
#             return selected_categories

#         except ValueError:
#             print("Invalid format. Please use comma-separated numbers (e.g., 1, 3, 5).")

# # --- USER INPUT LOGIC (Comprehensive with Skip Logic) ---

# def get_user_input_profile(feature_cols):
#     print("\n--- ENTER YOUR PROFILE DETAILS ---")
    
#     user_vector = {}
#     input_received_features = set()

#     # 1. Continuous Feature Inputs (CGPA and Age - MANDATORY)
#     print("\n[ STEP 1: ACADEMIC AND PERSONAL DATA (MANDATORY) ]")
    
#     cgpa_raw = safe_input("Enter your CGPA (0.0 to 10.0): ", is_float=True, min_val=0.0, max_val=10.0)
#     user_vector['CGPA_NORMALIZED'] = cgpa_raw / 10.0
#     input_received_features.add('CGPA_NORMALIZED')
    
#     age_raw = safe_input("Enter your Age (18 to 60): ", is_float=True, min_val=18, max_val=60)
#     user_vector['AGE_NORMALIZED'] = (age_raw - 18) / (60 - 18)
#     input_received_features.add('AGE_NORMALIZED')

#     # 2. Mandatory Binary Inputs 
#     print("\n[ STEP 2B: MANDATORY SKILLS & ATTRIBUTES (1=Yes, 0=No) ]")

#     for category, skills in MANDATORY_BINARY_MAPS.items():
#         print(f"\n-- CATEGORY: {category.replace(' (Mandatory)', '')} --")
        
#         for feature_name in skills:
#             feature_name_cleaned = clean_attribute_name(feature_name)
            
#             if feature_name_cleaned in feature_cols:
#                 prompt_name = feature_name.replace('_', ' ').title()
#                 user_vector[feature_name_cleaned] = safe_input(f"  {prompt_name}: (1/0) > ")
#                 input_received_features.add(feature_name_cleaned)

#     # 3. Differentiated/Skippable Binary Inputs (Technology Skills)
#     selected_tech_categories = select_categories(TECH_BINARY_MAPS)
#     print("\n[ STEP 2C: IN-DEMAND TECHNOLOGY SKILLS INPUT (1=Yes, 0=No) ]")

#     for category in selected_tech_categories:
#         skills = TECH_BINARY_MAPS[category]
#         print(f"\n-- CATEGORY: {category.replace(' (Tech)', '')} --")
        
#         for feature_name in skills:
#             feature_name_cleaned = clean_attribute_name(feature_name)
            
#             if feature_name_cleaned in feature_cols:
#                 prompt_name = feature_name.replace('_', ' ').title()
#                 user_vector[feature_name_cleaned] = safe_input(f"  {prompt_name}: (1/0) > ")
#                 input_received_features.add(feature_name_cleaned)

#     # 4. Default all unprompted/unmapped features (the SKIPPED Tech skills) to 0
#     defaulted_count = 0
    
#     # Combine all feature names from BOTH mandatory and tech maps
#     all_known_features = set()
#     for skills in MANDATORY_BINARY_MAPS.values():
#         all_known_features.update([clean_attribute_name(s) for s in skills])
#     for skills in TECH_BINARY_MAPS.values():
#         all_known_features.update([clean_attribute_name(s) for s in skills])
    
#     # Default all features that are known but were not prompted (the skipped tech skills)
#     for col in feature_cols:
#         if col not in input_received_features and col in all_known_features:
#             user_vector[col] = 0
#             defaulted_count += 1
#         elif col not in input_received_features and col not in ['Occupation_Code', 'Occupation_Name', 'CGPA_NORMALIZED', 'AGE_NORMALIZED']:
#              # Catch any stray feature not in maps (should be minimal)
#              user_vector[col] = 0
#              defaulted_count += 1
            
#     print(f"\n(Automatically set {defaulted_count} unselected/unmapped features to 0.)")

#     # Create DataFrame and ensure proper column order and shape
#     user_df = pd.DataFrame([user_vector])
#     user_df = user_df.reindex(columns=feature_cols, fill_value=0)
    
#     return user_df.iloc[0]

# # --- TRAINING AND RECOMMENDATION ---

# def train_career_recommender(data_file):
#     try:
#         df = pd.read_csv(data_file)
#     except FileNotFoundError:
#         print(f"Error: Training data file '{data_file}' not found. Please ensure data generation completed successfully.")
#         return None, None
        
#     # Add new continuous columns to the training data *before* getting features
#     if 'CGPA_NORMALIZED' not in df.columns:
#         df['CGPA_NORMALIZED'] = 0.5
#     if 'AGE_NORMALIZED' not in df.columns:
#         df['AGE_NORMALIZED'] = 0.5
        
#     feature_cols = [col for col in df.columns if col not in ['Occupation_Code', 'Occupation_Name']]
#     X = df[feature_cols]
    
#     # FIX: Fill NaN values with 0 and ensure numeric types
#     X = X.fillna(0).astype(float)
    
#     # *** FINAL FIX FOR MULTIPLE RECOMMENDATIONS: INCREASE n_neighbors to 50 ***
#     # This dramatically increases the search depth to find 4 unique jobs.
#     model = NearestNeighbors(n_neighbors=50, metric='cosine') 
#     model.fit(X)
    
#     return model, df, feature_cols

# def get_recommendations(model, df_train, user_vector, feature_cols, num_recommendations=4):
#     user_features = user_vector.to_numpy().reshape(1, -1)
    
#     distances, indices = model.kneighbors(user_features)
    
#     recommended_jobs = []
    
#     # Iterate through the 50 nearest neighbors and collect up to 4 unique jobs
#     for i in indices.flatten():
#         job_name = df_train.iloc[i]['Occupation_Name']
#         if job_name not in recommended_jobs:
#             recommended_jobs.append(job_name)
#             if len(recommended_jobs) >= num_recommendations:
#                 break
                
#     return recommended_jobs

# # --- DATA GENERATION (Placeholder) ---
# # NOTE: The full, complex data generation logic must be completed and working 
# # in your environment for the DATA_FILE to be accurate.
# def parse_and_generate_data(input_file, num_rows_per_job=500):
#     # This is a dummy function to ensure the main() logic runs without error 
#     # if the CSV is missing. You need the complete logic from earlier turns here.
#     COMPREHENSIVE_FEATURES = [
#         'STRUCTURED_QUERY_SQL', 'MICROSOFT_EXCEL', 'MICROSOFT_OFFICE', 'SPEAKING', 
#         'CRITICAL_THINKING', 'MATHEMATICS', 'ANALYTICAL_THINKING', 'ATTENTION_TO_DETAIL', 
#         'INVESTIGATIVE', 'CONVENTIONAL', 'ARTISTIC'
#     ]
#     df_combined = pd.DataFrame(columns=['Occupation_Code', 'Occupation_Name'] + COMPREHENSIVE_FEATURES)
#     return df_combined, {} 


# # --- MAIN EXECUTION ---

# def main():
#     DATA_FILE = "synthetic_20_job_binary_dataset.csv"
    
#     # 1. Data Preparation (Attempt to generate data if missing)
#     try:
#         pd.read_csv(DATA_FILE)
#     except FileNotFoundError:
#         print("Training data not found. Attempting minimal data generation to ensure structure...")
#         try:
#              df_train, _ = parse_and_generate_data(input_file="minor spread.xlsx - Sheet1.csv", num_rows_per_job=5)
#              df_train.to_csv(DATA_FILE, index=False)
#         except Exception as e:
#              print(f"Cannot proceed without training data. Error: {e}")
#              return
            
#     # 2. Train the Recommender
#     model, df_train, feature_cols = train_career_recommender(DATA_FILE)
#     if model is None:
#         return

#     # 3. Get User Input (Uses the full comprehensive input function with skip logic)
#     user_vector = get_user_input_profile(feature_cols)

#     # 4. Get Recommendations
#     recommendations = get_recommendations(model, df_train, user_vector, feature_cols, num_recommendations=4)
    
#     print("\n========================================")
#     print("TOP 4 CAREER RECOMMENDATIONS:")
#     print("========================================")
#     for i, job in enumerate(recommendations, 1):
#         print(f"{i}. {job}")
#     print("========================================")

# if __name__ == "__main__":
#     main()




# import pandas as pd
# import numpy as np
# import re
# from sklearn.ensemble import RandomForestClassifier # NEW MODEL
# from sklearn.preprocessing import LabelEncoder     # NEW FOR TARGET VARIABLE ENCODING
# from sklearn.neighbors import NearestNeighbors     # Kept for reference, but not used in the final model

# # --- STATIC MAPPINGS FOR ALL FEATURE CATEGORIES ---

# # 1. Differentiated/Skippable Binary Inputs (Technology Skills)
# TECH_BINARY_MAPS = {
#     'Analytical (Tech)': ['ADOBE', 'AHREFS_SITE_EXPLORER', 'AMAZON_REDSHIFT', 'BING_ADS', 'GOOGLE', 'GOOGLE_ADS', 'INFORMATICA', 'MICROSOFT_EXCEL', 'MICROSOFT_POWER_BI', 'MICROSOFT_SQL_SERVER_INTEGRATION_SERVICES_SSIS', 'MOZ_SEARCH_ENGINE_OPTIMIZATION_SEO', 'PANDAS', 'PYTORCH', 'R', 'SAS', 'SCIKIT_LEARN', 'SCREAMING_FROG_SEO_SPIDER', 'SEMRUSH', 'STRUCTURED_QUERY_SQL', 'TABLEAU', 'TENSORFLOW'],
#     'Creative (Tech)': ['ADOBE_AFTER_EFFECTS', 'ADOBE_CREATIVE_CLOUD', 'ADOBE_DREAMWEAVER', 'ADOBE_ILLUSTRATOR', 'ADOBE_INDESIGN', 'ADOBE_PHOTOSHOP', 'ADOBE_PREMIERE_PRO', 'AUTODESK_MAYA', 'CASCADING_STYLE_SHEETS_CSS', 'FIGMA', 'GOOGLE_ANGULAR', 'HYPERTEXT_MARKUP_LANGUAGE_HTML', 'JAVASCRIPT', 'MICROSOFT_DIRECTX', 'REACT', 'UNITY_TECHNOLOGIES_UNITY', 'UNREAL_TECHNOLOGY_UNREAL_ENGINE', 'VULKAN_GRAPHICS_API', 'WEB_APPLICATION', 'WORDPRESS'],
#     'Logical (Tech)': ['APACHE_HADOOP', 'APACHE_KAFKA', 'APACHE_SPARK', 'BASH', 'C', 'C#', 'C++', 'MICROSOFT_SQL_SERVER', 'MYSQL', 'NODE_JS', 'NOSQL', 'ORACLE_JAVA', 'ORACLE_PL_SQL', 'POSTGRESQL', 'PYTHON', 'RESTFUL_API', 'SCALA', 'TYPESCRIPT'],
#     'Practical (Tech)': ['AMAZON_WEB_SERVICES_AWS', 'ANSIBLE', 'CISCO', 'DOCKER', 'FIREWALL', 'GIT', 'IBM_TERRAFORM', 'JENKINS_CI', 'KUBERNETES', 'LINUX', 'MICROSOFT_ACTIVE_DIRECTORY', 'MICROSOFT_AZURE', 'MICROSOFT_POWERSHELL', 'MICROSOFT_SYSTEM_CENTER_CONFIGURATION_MANAGER', 'MICROSOFT_WINDOWS', 'MICROSOFT_WINDOWS_SERVER', 'SERVICENOW', 'OPERATING_SYSTEM_SOFTWARE', 'ORACLE_CLOUD', 'PHP', 'POSTMAN', 'RED_HAT_ENTERPRISE_LINUX', 'SALESFORCE', 'SAP', 'SELENIUM', 'UNIX'],
#     'Social (Tech)': ['ATLASSIAN_JIRA', 'GOOGLE', 'INFORMATION_TECHNICAL_PROJECT_MANAGER', 'MICROSOFT_OFFICE', 'MICROSOFT_OUTLOOK', 'MICROSOFT_POWERPOINT', 'MICROSOFT_SHAREPOINT', 'MICROSOFT_VISIO', 'SALES_ENGINEER', 'SCRUM_MASTER'],
# }

# # 2. Mandatory Binary Inputs (Comprehensive Skills, Knowledge, Work Styles, Interests, Education)
# MANDATORY_BINARY_MAPS = {
#     'Comprehensive Skills (Mandatory)': ['SPEAKING', 'READING_COMPREHENSION', 'ACTIVE_LISTENING', 'CRITICAL_THINKING', 'SYSTEMS_ANALYSIS', 'SYSTEMS_EVALUATION', 'ACTIVE_LEARNING', 'JUDGMENT_DECISION_MAKING', 'COMPLEX_PROBLEM_SOLVING', 'MONITORING', 'WRITING'],
#     'Knowledge (Mandatory)': ['COMPUTERS_ELECTRONICS', 'CUSTOMER_PERSONAL_SERVICE', 'ENGLISH_LANGUAGE', 'MATHEMATICS', 'ADMINISTRATION_MANAGEMENT', 'EDUCATION_TRAINING', 'ECONOMICS_ACCOUNTING', 'LAW_GOVERNMENT', 'SALES_MARKETING', 'HUMAN_RESOURCES', 'THERAPY_COUNSELING', 'PUBLIC_SAFETY_SECURITY', 'DESIGN', 'FINE_ARTS', 'COMMUNICATIONS_MEDIA'],
#     'Work Styles (Mandatory)': ['ANALYTICAL_THINKING', 'ATTENTION_TO_DETAIL', 'DEPENDABILITY', 'COOPERATION', 'INTEGRITY', 'INITIATIVE', 'PERSISTENCE', 'INNOVATION', 'STRESS_TOLERANCE', 'ADAPTABILITY_FLEXIBILITY', 'INDEPENDENCE', 'ACHIEVEMENT_EFFORT', 'CONCERN_FOR_OTHERS', 'SELF_CONTROL', 'LEADERSHIP'],
#     'Interests (Mandatory)': ['INVESTIGATIVE', 'CONVENTIONAL', 'ARTISTIC'],
#     'Education (Mandatory)': ['B_TECH', 'M_TECH'],
# }


# # --- UTILITY FUNCTIONS ---

# def clean_attribute_name(attribute_str):
#     """Standard cleaning logic to match features in the training data."""
#     # Removes common words and replaces non-alphanumeric characters with underscores
#     clean_attribute = re.sub(r'(software|and|the|language|designers|for|occupation|making|training|personal|service)', '', attribute_str, flags=re.IGNORECASE).strip()
#     clean_attribute = re.sub(r'[\s\W]+', '_', clean_attribute).strip('_').upper()
#     return clean_attribute

# def safe_input(prompt, is_float=False, min_val=0, max_val=1):
#     """Handles input validation for binary or continuous data."""
#     while True:
#         try:
#             value = input(prompt)
#             if is_float:
#                 val = float(value)
#                 if min_val <= val <= max_val:
#                     return val
#                 else:
#                     print(f"Input must be between {min_val} and {max_val}.")
#             else:
#                 val = int(value)
#                 if val in [0, 1]:
#                     return val
#                 else:
#                     print("Invalid input. Please enter 1 or 0.")
#         except ValueError:
#             print("Invalid input. Please enter a valid number.")

# def select_categories(category_maps):
#     """Presents a menu for the user to select which categories they want to input."""
    
#     print("\n[ STEP 2A: SELECT IN-DEMAND TECHNOLOGY SKILL CATEGORIES TO ENTER ]")
#     print("Please select the categories you want to enter skills/attributes for. (e.g., 1, 3)")
    
#     category_names = list(category_maps.keys())
#     for i, name in enumerate(category_names):
#         print(f"  {i+1}. {name.replace(' (Tech)', '')}")
        
#     while True:
#         selection_input = input("\nEnter the numbers of the Tech categories you want to input (comma-separated): > ")
#         try:
#             # Parse input and convert to 0-indexed list
#             selected_indices = [int(x.strip()) - 1 for x in selection_input.split(',')]
            
#             # Validate indices
#             valid_indices = [i for i in selected_indices if 0 <= i < len(category_names)]
            
#             if not valid_indices:
#                 print("No valid categories selected. Please enter at least one or try again.")
#                 continue

#             selected_categories = [category_names[i] for i in valid_indices]
#             print(f"\n✅ You chose to input for: {', '.join([c.replace(' (Tech)', '') for c in selected_categories])}")
#             return selected_categories

#         except ValueError:
#             print("Invalid format. Please use comma-separated numbers (e.g., 1, 3, 5).")

# # --- USER INPUT LOGIC (Comprehensive with Skip Logic) ---

# def get_user_input_profile(feature_cols):
#     """Gathers input from the user for all relevant features."""
#     print("\n--- ENTER YOUR PROFILE DETAILS ---")
    
#     user_vector = {}
#     input_received_features = set()

#     # 1. Continuous Feature Inputs (CGPA and Age - MANDATORY)
#     print("\n[ STEP 1: ACADEMIC AND PERSONAL DATA (MANDATORY) ]")
    
#     # Normalized CGPA (0.0 to 1.0)
#     cgpa_raw = safe_input("Enter your CGPA (0.0 to 10.0): ", is_float=True, min_val=0.0, max_val=10.0)
#     user_vector['CGPA_NORMALIZED'] = cgpa_raw / 10.0
#     input_received_features.add('CGPA_NORMALIZED')
    
#     # Normalized Age (0.0 to 1.0)
#     age_raw = safe_input("Enter your Age (18 to 60): ", is_float=True, min_val=18, max_val=60)
#     user_vector['AGE_NORMALIZED'] = (age_raw - 18) / (60 - 18)
#     input_received_features.add('AGE_NORMALIZED')

#     # 2. Mandatory Binary Inputs 
#     print("\n[ STEP 2B: MANDATORY SKILLS & ATTRIBUTES (1=Yes, 0=No) ]")

#     for category, skills in MANDATORY_BINARY_MAPS.items():
#         print(f"\n-- CATEGORY: {category.replace(' (Mandatory)', '')} --")
        
#         for feature_name in skills:
#             feature_name_cleaned = clean_attribute_name(feature_name)
            
#             if feature_name_cleaned in feature_cols:
#                 prompt_name = feature_name.replace('_', ' ').title()
#                 user_vector[feature_name_cleaned] = safe_input(f"  {prompt_name}: (1/0) > ")
#                 input_received_features.add(feature_name_cleaned)

#     # 3. Differentiated/Skippable Binary Inputs (Technology Skills)
#     selected_tech_categories = select_categories(TECH_BINARY_MAPS)
#     print("\n[ STEP 2C: IN-DEMAND TECHNOLOGY SKILLS INPUT (1=Yes, 0=No) ]")

#     for category in selected_tech_categories:
#         skills = TECH_BINARY_MAPS[category]
#         print(f"\n-- CATEGORY: {category.replace(' (Tech)', '')} --")
        
#         for feature_name in skills:
#             feature_name_cleaned = clean_attribute_name(feature_name)
            
#             if feature_name_cleaned in feature_cols:
#                 prompt_name = feature_name.replace('_', ' ').title()
#                 user_vector[feature_name_cleaned] = safe_input(f"  {prompt_name}: (1/0) > ")
#                 input_received_features.add(feature_name_cleaned)

#     # 4. Default all unprompted/unmapped features (the SKIPPED Tech skills) to 0
#     defaulted_count = 0
    
#     all_known_features = set()
#     for skills in MANDATORY_BINARY_MAPS.values():
#         all_known_features.update([clean_attribute_name(s) for s in skills])
#     for skills in TECH_BINARY_MAPS.values():
#         all_known_features.update([clean_attribute_name(s) for s in skills])
    
#     # Default all features that are known but were not prompted (e.g., skipped tech skills)
#     for col in feature_cols:
#         if col not in input_received_features and col in all_known_features:
#             user_vector[col] = 0
#             defaulted_count += 1
#         elif col not in input_received_features and col not in ['Occupation_Code', 'Occupation_Name', 'CGPA_NORMALIZED', 'AGE_NORMALIZED']:
#              # Catch any stray feature not in maps (should be minimal)
#              user_vector[col] = 0
#              defaulted_count += 1
            
#     print(f"\n(Automatically set {defaulted_count} unselected/unmapped features to 0.)")

#     # Create DataFrame and ensure proper column order and shape
#     user_df = pd.DataFrame([user_vector])
#     user_df = user_df.reindex(columns=feature_cols, fill_value=0)
    
#     return user_df.iloc[0]


# # --- DATA GENERATION (Placeholder to ensure main() runs) ---
# # NOTE: This function is a placeholder based on your provided code structure.
# # For a real application, you must replace this with the complete data generation 
# # logic for all 20 jobs to create a functional multi-class training dataset.
# def parse_and_generate_data(input_file, num_rows_per_job=500):
#     COMPREHENSIVE_FEATURES = [
#         'STRUCTURED_QUERY_SQL', 'MICROSOFT_EXCEL', 'MICROSOFT_OFFICE', 'SPEAKING', 
#         'CRITICAL_THINKING', 'MATHEMATICS', 'ANALYTICAL_THINKING', 'ATTENTION_TO_DETAIL', 
#         'INVESTIGATIVE', 'CONVENTIONAL', 'ARTISTIC'
#     ]
#     # Creates an empty DataFrame with necessary columns
#     df_combined = pd.DataFrame(columns=['Occupation_Code', 'Occupation_Name'] + COMPREHENSIVE_FEATURES + ['CGPA_NORMALIZED', 'AGE_NORMALIZED'])
#     return df_combined, {} 


# # --- TRAINING AND RECOMMENDATION (RANDOM FOREST) ---

# def train_career_recommender(data_file):
#     """Loads data, prepares X and Y, and trains the RandomForestClassifier."""
#     try:
#         df = pd.read_csv(data_file)
#     except FileNotFoundError:
#         print(f"Error: Training data file '{data_file}' not found. Please ensure your 20-job binary data generation completed successfully.")
#         return None, None, None, None

#     # Add new continuous columns to the training data *before* getting features
#     if 'CGPA_NORMALIZED' not in df.columns:
#         df['CGPA_NORMALIZED'] = 0.5
#     if 'AGE_NORMALIZED' not in df.columns:
#         df['AGE_NORMALIZED'] = 0.5
        
#     feature_cols = [col for col in df.columns if col not in ['Occupation_Code', 'Occupation_Name']]
#     X = df[feature_cols].fillna(0).astype(float) # Features (User Input attributes)
    
#     # 1. Prepare Target Variable (Y) for Classification
#     Y_raw = df['Occupation_Name'] # Target variable (Job Name)
    
#     # Initialize LabelEncoder to convert job names into numerical labels
#     label_encoder = LabelEncoder()
#     Y = label_encoder.fit_transform(Y_raw)
    
#     # 2. Train Random Forest Classifier
#     # n_estimators=100 is a good default number of trees
#     model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
#     model.fit(X, Y)
    
#     # Returns the trained model, the full training data, feature list, and the encoder
#     return model, df, feature_cols, label_encoder

# def get_recommendations(model, df_train, user_vector, feature_cols, label_encoder, num_recommendations=4):
#     """Uses the trained Random Forest model to get probabilistic recommendations."""
    
#     user_features = user_vector.to_numpy().reshape(1, -1)
    
#     # Use predict_proba to get the probability score for every job class
#     # The output is an array of probabilities for each class
#     probabilities = model.predict_proba(user_features)[0]
    
#     # Get the indices of the top N probabilities (descending order)
#     # np.argsort returns indices in ascending order, so [::-1] reverses it
#     top_n_indices = np.argsort(probabilities)[::-1][:num_recommendations]
    
#     # Map the numerical class indices back to the original job names
#     recommended_jobs = label_encoder.inverse_transform(top_n_indices)
                
#     return recommended_jobs.tolist()


# # --- MAIN EXECUTION ---

# def main():
#     # REQUIRED: This file must contain binary features for multiple jobs (e.g., 20)
#     DATA_FILE = "synthetic_20_job_binary_dataset.csv"
    
#     # 1. Data Preparation (Attempt to generate data if missing)
#     try:
#         pd.read_csv(DATA_FILE)
#     except FileNotFoundError:
#         print(f"Training data file '{DATA_FILE}' not found.")
#         print("Running placeholder data generation to provide minimum column structure...")
#         try:
#              # This will likely create an empty file but allows the program to proceed to training setup
#              df_train, _ = parse_and_generate_data(input_file="minor spread.xlsx - Sheet1.csv", num_rows_per_job=5)
#              df_train.to_csv(DATA_FILE, index=False)
#              print("Placeholder CSV created. Rerunning training attempt...")
#         except Exception as e:
#              print(f"Cannot create placeholder training data structure. Error: {e}")
#              return
            
#     # 2. Train the Recommender (Model, DataFrame, Features, Encoder)
#     results = train_career_recommender(DATA_FILE)
    
#     if results is None or results[0] is None:
#         print("Training failed or model could not be initialized. Check your data file.")
#         return
        
#     model, df_train, feature_cols, label_encoder = results

#     # 3. Get User Input 
#     user_vector = get_user_input_profile(feature_cols)

#     # 4. Get Recommendations
#     recommendations = get_recommendations(model, df_train, user_vector, feature_cols, label_encoder, num_recommendations=4)
    
#     print("\n========================================")
#     print("TOP 4 CAREER RECOMMENDATIONS (Random Forest):")
#     print("========================================")
#     for i, job in enumerate(recommendations, 1):
#         print(f"{i}. {job}")
#     print("========================================")

# if __name__ == "__main__":
#     main()





import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier              # <-- XGBoost Model
from sklearn.model_selection import train_test_split # <-- For accuracy measurement
from sklearn.metrics import accuracy_score       # <-- For accuracy measurement
import sys
# Removed unused import: from sklearn.neighbors import NearestNeighbors

# --- STATIC MAPPINGS FOR ALL FEATURE CATEGORIES ---

# 1. Differentiated/Skippable Binary Inputs (Technology Skills)
TECH_BINARY_MAPS = {
    'Analytical (Tech)': ['ADOBE', 'AHREFS_SITE_EXPLORER', 'AMAZON_REDSHIFT', 'BING_ADS', 'GOOGLE', 'GOOGLE_ADS', 'INFORMATICA', 'MICROSOFT_EXCEL', 'MICROSOFT_POWER_BI', 'MICROSOFT_SQL_SERVER_INTEGRATION_SERVICES_SSIS', 'MOZ_SEARCH_ENGINE_OPTIMIZATION_SEO', 'PANDAS', 'PYTORCH', 'R', 'SAS', 'SCIKIT_LEARN', 'SCREAMING_FROG_SEO_SPIDER', 'SEMRUSH', 'STRUCTURED_QUERY_SQL', 'TABLEAU', 'TENSORFLOW'],
    'Creative (Tech)': ['ADOBE_AFTER_EFFECTS', 'ADOBE_CREATIVE_CLOUD', 'ADOBE_DREAMWEAVER', 'ADOBE_ILLUSTRATOR', 'ADOBE_INDESIGN', 'ADOBE_PHOTOSHOP', 'ADOBE_PREMIERE_PRO', 'AUTODESK_MAYA', 'CASCADING_STYLE_SHEETS_CSS', 'FIGMA', 'GOOGLE_ANGULAR', 'HYPERTEXT_MARKUP_LANGUAGE_HTML', 'JAVASCRIPT', 'MICROSOFT_DIRECTX', 'REACT', 'UNITY_TECHNOLOGIES_UNITY', 'UNREAL_TECHNOLOGY_UNREAL_ENGINE', 'VULKAN_GRAPHICS_API', 'WEB_APPLICATION', 'WORDPRESS'],
    'Logical (Tech)': ['APACHE_HADOOP', 'APACHE_KAFKA', 'APACHE_SPARK', 'BASH', 'C', 'C#', 'C++', 'MICROSOFT_SQL_SERVER', 'MYSQL', 'NODE_JS', 'NOSQL', 'ORACLE_JAVA', 'ORACLE_PL_SQL', 'POSTGRESQL', 'PYTHON', 'RESTFUL_API', 'SCALA', 'TYPESCRIPT'],
    'Practical (Tech)': ['AMAZON_WEB_SERVICES_AWS', 'ANSIBLE', 'CISCO', 'DOCKER', 'FIREWALL', 'GIT', 'IBM_TERRAFORM', 'JENKINS_CI', 'KUBERNETES', 'LINUX', 'MICROSOFT_ACTIVE_DIRECTORY', 'MICROSOFT_AZURE', 'MICROSOFT_POWERSHELL', 'MICROSOFT_SYSTEM_CENTER_CONFIGURATION_MANAGER', 'MICROSOFT_WINDOWS', 'MICROSOFT_WINDOWS_SERVER', 'SERVICENOW', 'OPERATING_SYSTEM_SOFTWARE', 'ORACLE_CLOUD', 'PHP', 'POSTMAN', 'RED_HAT_ENTERPRISE_LINUX', 'SALESFORCE', 'SAP', 'SELENIUM', 'UNIX'],
    'Social (Tech)': ['ATLASSIAN_JIRA', 'GOOGLE', 'INFORMATION_TECHNICAL_PROJECT_MANAGER', 'MICROSOFT_OFFICE', 'MICROSOFT_OUTLOOK', 'MICROSOFT_POWERPOINT', 'MICROSOFT_SHAREPOINT', 'MICROSOFT_VISIO', 'SALES_ENGINEER', 'SCRUM_MASTER'],
}

# 2. Mandatory Binary Inputs (Comprehensive Skills, Knowledge, Work Styles, Interests, Education)
MANDATORY_BINARY_MAPS = {
    'Comprehensive Skills (Mandatory)': ['SPEAKING', 'READING_COMPREHENSION', 'ACTIVE_LISTENING', 'CRITICAL_THINKING', 'SYSTEMS_ANALYSIS', 'SYSTEMS_EVALUATION', 'ACTIVE_LEARNING', 'JUDGMENT_DECISION_MAKING', 'COMPLEX_PROBLEM_SOLVING', 'MONITORING', 'WRITING'],
    'Knowledge (Mandatory)': ['COMPUTERS_ELECTRONICS', 'CUSTOMER_PERSONAL_SERVICE', 'ENGLISH_LANGUAGE', 'MATHEMATICS', 'ADMINISTRATION_MANAGEMENT', 'EDUCATION_TRAINING', 'ECONOMICS_ACCOUNTING', 'LAW_GOVERNMENT', 'SALES_MARKETING', 'HUMAN_RESOURCES', 'THERAPY_COUNSELING', 'PUBLIC_SAFETY_SECURITY', 'DESIGN', 'FINE_ARTS', 'COMMUNICATIONS_MEDIA'],
    'Work Styles (Mandatory)': ['ANALYTICAL_THINKING', 'ATTENTION_TO_DETAIL', 'DEPENDABILITY', 'COOPERATION', 'INTEGRITY', 'INITIATIVE', 'PERSISTENCE', 'INNOVATION', 'STRESS_TOLERANCE', 'ADAPTABILITY_FLEXIBILITY', 'INDEPENDENCE', 'ACHIEVEMENT_EFFORT', 'CONCERN_FOR_OTHERS', 'SELF_CONTROL', 'LEADERSHIP'],
    'Interests (Mandatory)': ['INVESTIGATIVE', 'CONVENTIONAL', 'ARTISTIC'],
    'Education (Mandatory)': ['B_TECH', 'M_TECH'],
}


# --- UTILITY FUNCTIONS ---

def clean_attribute_name(attribute_str):
    """Standard cleaning logic to match features in the training data."""
    clean_attribute = re.sub(r'(software|and|the|language|designers|for|occupation|making|training|personal|service)', '', attribute_str, flags=re.IGNORECASE).strip()
    clean_attribute = re.sub(r'[\s\W]+', '_', clean_attribute).strip('_').upper()
    return clean_attribute

def safe_input(prompt, is_float=False, min_val=0, max_val=1):
    """Handles input validation for binary or continuous data."""
    while True:
        try:
            value = input(prompt)
            if is_float:
                val = float(value)
                if min_val <= val <= max_val:
                    return val
                else:
                    print(f"Input must be between {min_val} and {max_val}.")
            else:
                val = int(value)
                if val in [0, 1]:
                    return val
                else:
                    print("Invalid input. Please enter 1 or 0.")
        except ValueError:
            print("Invalid input. Please enter a valid number.")

def select_categories(category_maps):
    """Presents a menu for the user to select which categories they want to input."""
    
    print("\n[ STEP 2A: SELECT IN-DEMAND TECHNOLOGY SKILL CATEGORIES TO ENTER ]")
    print("Please select the categories you want to enter skills/attributes for. (e.g., 1, 3)")
    
    category_names = list(category_maps.keys())
    for i, name in enumerate(category_names):
        print(f"  {i+1}. {name.replace(' (Tech)', '')}")
        
    while True:
        selection_input = input("\nEnter the numbers of the Tech categories you want to input (comma-separated): > ")
        try:
            selected_indices = [int(x.strip()) - 1 for x in selection_input.split(',')]
            valid_indices = [i for i in selected_indices if 0 <= i < len(category_names)]
            
            if not valid_indices:
                print("No valid categories selected. Please enter at least one or try again.")
                continue

            selected_categories = [category_names[i] for i in valid_indices]
            print(f"\n✅ You chose to input for: {', '.join([c.replace(' (Tech)', '') for c in selected_categories])}")
            return selected_categories

        except ValueError:
            print("Invalid format. Please use comma-separated numbers (e.g., 1, 3, 5).")

# --- USER INPUT LOGIC (Comprehensive with Skip Logic) ---

def get_user_input_profile(feature_cols):
    """Gathers input from the user for all relevant features."""
    print("\n--- ENTER YOUR PROFILE DETAILS ---")
    
    user_vector = {}
    input_received_features = set()

    # 1. Continuous Feature Inputs (CGPA and Age - MANDATORY)
    print("\n[ STEP 1: ACADEMIC AND PERSONAL DATA (MANDATORY) ]")
    
    # Normalized CGPA (0.0 to 1.0)
    cgpa_raw = safe_input("Enter your CGPA (0.0 to 10.0): ", is_float=True, min_val=0.0, max_val=10.0)
    user_vector['CGPA_NORMALIZED'] = cgpa_raw / 10.0
    input_received_features.add('CGPA_NORMALIZED')
    
    # Normalized Age (0.0 to 1.0)
    age_raw = safe_input("Enter your Age (18 to 60): ", is_float=True, min_val=18, max_val=60)
    user_vector['AGE_NORMALIZED'] = (age_raw - 18) / (60 - 18)
    input_received_features.add('AGE_NORMALIZED')

    # 2. Mandatory Binary Inputs 
    print("\n[ STEP 2B: MANDATORY SKILLS & ATTRIBUTES (1=Yes, 0=No) ]")

    for category, skills in MANDATORY_BINARY_MAPS.items():
        print(f"\n-- CATEGORY: {category.replace(' (Mandatory)', '')} --")
        
        for feature_name in skills:
            feature_name_cleaned = clean_attribute_name(feature_name)
            
            if feature_name_cleaned in feature_cols:
                prompt_name = feature_name.replace('_', ' ').title()
                user_vector[feature_name_cleaned] = safe_input(f"  {prompt_name}: (1/0) > ")
                input_received_features.add(feature_name_cleaned)

    # 3. Differentiated/Skippable Binary Inputs (Technology Skills)
    selected_tech_categories = select_categories(TECH_BINARY_MAPS)
    print("\n[ STEP 2C: IN-DEMAND TECHNOLOGY SKILLS INPUT (1=Yes, 0=No) ]")

    for category in selected_tech_categories:
        skills = TECH_BINARY_MAPS[category]
        print(f"\n-- CATEGORY: {category.replace(' (Tech)', '')} --")
        
        for feature_name in skills:
            feature_name_cleaned = clean_attribute_name(feature_name)
            
            if feature_name_cleaned in feature_cols:
                prompt_name = feature_name.replace('_', ' ').title()
                user_vector[feature_name_cleaned] = safe_input(f"  {prompt_name}: (1/0) > ")
                input_received_features.add(feature_name_cleaned)

    # 4. Default all unprompted/unmapped features (the SKIPPED Tech skills) to 0
    all_known_features = set()
    for skills in MANDATORY_BINARY_MAPS.values():
        all_known_features.update([clean_attribute_name(s) for s in skills])
    for skills in TECH_BINARY_MAPS.values():
        all_known_features.update([clean_attribute_name(s) for s in skills])
    
    defaulted_count = 0
    for col in feature_cols:
        if col not in input_received_features:
            user_vector[col] = 0
            defaulted_count += 1
            
    print(f"\n(Automatically set {defaulted_count} unselected/unmapped features to 0.)")

    # Create DataFrame and ensure proper column order and shape
    user_df = pd.DataFrame([user_vector])
    user_df = user_df.reindex(columns=feature_cols, fill_value=0)
    
    return user_df.iloc[0]

# --- DATA GENERATION (Placeholder) ---
def parse_and_generate_data(input_file, num_rows_per_job=500):
    COMPREHENSIVE_FEATURES = [
        'STRUCTURED_QUERY_SQL', 'MICROSOFT_EXCEL', 'MICROSOFT_OFFICE', 'SPEAKING', 
        'CRITICAL_THINKING', 'MATHEMATICS', 'ANALYTICAL_THINKING', 'ATTENTION_TO_DETAIL', 
        'INVESTIGATIVE', 'CONVENTIONAL', 'ARTISTIC'
    ]
    df_combined = pd.DataFrame(columns=['Occupation_Code', 'Occupation_Name'] + COMPREHENSIVE_FEATURES + ['CGPA_NORMALIZED', 'AGE_NORMALIZED'])
    return df_combined, {} 

def create_dummy_data(output_file, num_rows=100):
    """Creates dummy data for demonstration if the real data file is missing."""
    print(f"Generating DUMMY data for {output_file}. WARNING: Model accuracy will be random.")
    
    all_tech_features = [clean_attribute_name(s) for skills in TECH_BINARY_MAPS.values() for s in skills]
    all_mandatory_features = [clean_attribute_name(s) for skills in MANDATORY_BINARY_MAPS.values() for s in skills]
    all_features = all_tech_features + all_mandatory_features

    job_titles = ['Data Scientist', 'Software Engineer', 'UX Designer', 'Technical Writer']
    
    data = []
    for i in range(num_rows):
        row = {
            'Occupation_Code': f'DUMMY-{i}',
            'Occupation_Name': np.random.choice(job_titles),
            'CGPA_NORMALIZED': np.random.uniform(0.5, 1.0),
            'AGE_NORMALIZED': np.random.uniform(0.2, 0.5)
        }
        for feature in all_features:
            row[feature] = np.random.randint(0, 2)
        data.append(row)

    df_dummy = pd.DataFrame(data)
    df_dummy.to_csv(output_file, index=False)
    return df_dummy

# --- TRAINING AND RECOMMENDATION (XGBOOST IMPLEMENTATION) ---

def train_career_recommender(data_file):
    """Loads data, evaluates accuracy, and trains the final XGBClassifier model."""
    try:
        df = pd.read_csv(data_file)
    except FileNotFoundError:
        print(f"Error: Training data file '{data_file}' not found.")
        return None, None, None, None

    # Ensure required normalized columns exist
    if 'CGPA_NORMALIZED' not in df.columns: df['CGPA_NORMALIZED'] = 0.5
    if 'AGE_NORMALIZED' not in df.columns: df['AGE_NORMALIZED'] = 0.5
        
    feature_cols = [col for col in df.columns if col not in ['Occupation_Code', 'Occupation_Name']]
    X = df[feature_cols].fillna(0).astype(float) # Features
    
    # 1. Prepare Target Variable (Y) for Classification
    Y_raw = df['Occupation_Name'] 
    label_encoder = LabelEncoder()
    Y = label_encoder.fit_transform(Y_raw) # Converts job names to numerical labels 0, 1, 2...
    
    # 2. Splitting Data for Accuracy Measurement
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, 
        test_size=0.2, 
        random_state=42, 
        stratify=Y      
    )
    
    # 3. Train XGBoost Model (on training subset for evaluation)
    model = XGBClassifier(
        objective='multi:softprob', 
        n_estimators=200,          
        learning_rate=0.05,        
        random_state=42, 
        # REMOVED: use_label_encoder=False
        eval_metric='mlogloss',    
        n_jobs=-1
    )
    
    # Train on the training subset
    model.fit(X_train, Y_train)
    
    # 4. Evaluate Accuracy on the Test Set
    Y_pred = model.predict(X_test)
    accuracy = accuracy_score(Y_test, Y_pred)
    
    print("\n--- MODEL PERFORMANCE ---")
    print(f"XGBoost Model Accuracy on Test Set (20%): {accuracy:.4f}")
    print("-------------------------")
    
    # 5. Retrain the Final Model on the ENTIRE dataset
    final_model = XGBClassifier(
        objective='multi:softprob', 
        n_estimators=200, 
        learning_rate=0.05, 
        random_state=42, 
        # REMOVED: use_label_encoder=False
        eval_metric='mlogloss',
        n_jobs=-1
    )
    final_model.fit(X, Y)
    
    return final_model, df, feature_cols, label_encoder

def get_recommendations(model, user_vector, label_encoder, num_recommendations=4):
    """Uses the trained XGBoost model to get probabilistic recommendations."""
    
    user_features = user_vector.to_numpy().reshape(1, -1)
    
    # Use predict_proba to get the probability score for every job class
    probabilities = model.predict_proba(user_features)[0]
    
    # Get the indices of the top N probabilities (descending order)
    top_n_indices = np.argsort(probabilities)[::-1][:num_recommendations]
    
    # Map the numerical class indices back to the original job names
    recommended_jobs = label_encoder.inverse_transform(top_n_indices)
                
    return recommended_jobs.tolist()


# --- MAIN EXECUTION ---

def main():
    DATA_FILE = "synthetic_20_job_binary_dataset.csv"
    
    # 1. Data Preparation (Check for data file)
    try:
        pd.read_csv(DATA_FILE)
    except FileNotFoundError:
        print(f"WARNING: Training data file '{DATA_FILE}' not found.")
        print("Creating DUMMY data for demonstration purposes. Accuracy will be random.")
        create_dummy_data(DATA_FILE, num_rows=2000)
        
    # 2. Train the Recommender (Includes Accuracy Calculation)
    results = train_career_recommender(DATA_FILE)
    
    if results is None or results[0] is None:
        print("Training failed or model could not be initialized. Check your data file.")
        return
        
    model, df_train, feature_cols, label_encoder = results

    # 3. Get User Input 
    # Logic to handle both interactive and non-interactive environments
    if not sys.stdin.isatty():
        print("Non-interactive session detected. Skipping user input.")
        # Create a sample user vector to continue the process
        user_vector_data = {col: np.random.randint(0, 2) for col in feature_cols}
        user_vector_data['CGPA_NORMALIZED'] = 0.8
        user_vector_data['AGE_NORMALIZED'] = 0.4
        user_vector = pd.Series(user_vector_data, index=feature_cols)
    else:
        user_vector = get_user_input_profile(feature_cols)


    # 4. Get Recommendations
    recommendations = get_recommendations(model, user_vector, label_encoder, num_recommendations=4)
    
    print("\n========================================")
    print("TOP 4 CAREER RECOMMENDATIONS (XGBoost):")
    print("========================================")
    for i, job in enumerate(recommendations, 1):
        print(f"{i}. {job}")
    print("========================================")

if __name__ == "__main__":
    main()