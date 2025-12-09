import pandas as pd
import numpy as np
import re

def parse_and_generate_data(input_file, num_rows_per_job=500):
    """
    Parses the complex multi-table Excel file and generates synthetic binary data 
    for all contained job profiles based on a comprehensive list of 20 jobs.
    """
    # Using pd.read_excel as the file is in .xlsx format
    # Assuming your data is on the first sheet (sheet_name=0) and has no header (header=None).
    df = pd.read_excel(input_file, header=None, sheet_name=0) 
    
    df.columns = ['Col0', 'Col1', 'Col2'] 
    
    # Clean the data for robust parsing
    df_clean = df.astype(str).apply(lambda x: x.str.strip().str.replace(r'\s+', ' ', regex=True).str.lower())

    # Comprehensive map of all 20 job identifiers found/inferred in the spreadsheet
    JOB_LABELS = {
        "computer system analyst": "15-1211.00 - Computer Systems Analysts",
        "video game designers": "15-1255.01 - Video Game Designers",
        "graphic designers": "27-1024.00 - Graphic Designers",
        "13-1161.00 - market research analysts and marketing specialists": "13-1161.00 - Market Research Analysts and Marketing Specialists",
        "13-1161.01 - search marketing strategists": "13-1161.01 - Search Marketing Strategists",
        "13-2051.00 - financial and investment analysts": "13-2051.00 - Financial and Investment Analysts",
        "13-2099.01 - financial quantitative analysts": "13-2099.01 - Financial Quantitative Analysts",
        "15-1232.00 - computer user support specialists": "15-1232.00 - Computer User Support Specialists",
        "15-1242.00 - database administrators": "15-1242.00 - Database Administrators",
        "15-1243.00 - database architects": "15-1243.00 - Database Architects",
        "15-1244.00 - network and computer systems administrators": "15-1244.00 - Network and Computer Systems Administrators",
        "15-1252.00 - software developers": "15-1252.00 - Software Developers",
        "15-1253.00 - software quality assurance analysts and testers": "15-1253.00 - Software Quality Assurance Analysts and Testers",
        "15-1254.00 - web developers": "15-1254.00 - Web Developers",
        "15-1255.00 - web and digital interface designers": "15-1255.00 - Web and Digital Interface Designers",
        "15-1299.08 - computer systems engineers/architects": "15-1299.08 - Computer Systems Engineers/Architects",
        "15-1299.09 - information technology project managers": "15-1299.09 - Information Technology Project Managers",
        "15-2051.00 - data scientists": "15-2051.00 - Data Scientists",
        "15-2051.01 - business intelligence analysts": "15-2051.01 - Business Intelligence Analysts",
        "41-9031.00 - sales engineers": "41-9031.00 - Sales Engineers",
    }
    
    # Add reverse mapping for robust matching
    all_job_keys = set(JOB_LABELS.keys())
    for v in JOB_LABELS.values():
        code = v.split(' - ')[0]
        name = v.split(' - ')[1]
        all_job_keys.add(code.lower())
        all_job_keys.add(name.lower())
    
    SECTION_MAP = {
        "technology skill": "Tech_Skills",
        "skill": "Skills",
        "knowledge": "Knowledge",
        "interest": "Interests",
        "work style": "Work_Styles"
    }

    job_distributions = {}
    current_job = None
    current_section = None

    # --- Parsing Loop ---
    for _, row in df_clean.iterrows():
        # 1. Identify current job profile
        found_job = False
        for raw_job_key in all_job_keys:
            if raw_job_key in row['Col0'] or raw_job_key in row['Col1']:
                current_job = JOB_LABELS.get(raw_job_key, raw_job_key)
                
                if re.match(r'\d{2}-\d{4}\.\d{2}', current_job):
                    for full_label in JOB_LABELS.values():
                        if current_job in full_label:
                            current_job = full_label
                            break

                if current_job not in job_distributions:
                    job_distributions[current_job] = {}
                current_section = None
                found_job = True
                break
        if found_job: continue 

        # 2. Identify current section
        for raw_section, clean_key in SECTION_MAP.items():
            if raw_section in row['Col0'] or raw_section in row['Col1']:
                current_section = clean_key
                break
        
        # 3. Extract data: rely solely on Col0 being a number (score/percentage)
        if current_job and current_section:
            value_str = row['Col0']
            attribute_str = row['Col1']
            
            if re.match(r'^\d+(\.\d+)?$', value_str):
                try:
                    value = float(value_str)
                    probability = value / 100.0
                    
                    if current_section not in job_distributions[current_job]:
                        job_distributions[current_job][current_section] = {}
                    
                    # Clean up attribute name for column header
                    clean_attribute = re.sub(r'(software|and|the|language|designers|for|occupation|making|training|personal|service)', '', attribute_str).strip()
                    clean_attribute = re.sub(r'[\s\W]+', '_', clean_attribute).strip('_').upper()

                    job_distributions[current_job][current_section][clean_attribute] = probability
                    
                except ValueError:
                    pass

    # --- Generation Loop ---
    all_data_frames = []
    
    for job_label, job_data in job_distributions.items():
        if not job_data: continue
        
        all_probs = {}
        for section_data in job_data.values():
            all_probs.update(section_data)
        
        match = re.search(r'(\d{2}-\d{4}\.\d{2})\s*-\s*(.*)', job_label, re.IGNORECASE)
        occ_code, occ_name = match.groups() if match else ('UNKNOWN_CODE', job_label)

        rows = []
        for _ in range(num_rows_per_job):
            row = {
                "Occupation_Code": occ_code.strip().upper(),
                "Occupation_Name": occ_name.strip().title(),
            }

            for attr, prob in all_probs.items():
                row[attr] = np.random.choice([1, 0], p=[prob, 1 - prob])
            
            rows.append(row)
            
        df_job = pd.DataFrame(rows)
        all_data_frames.append(df_job)

    # Concatenate all DataFrames and return
    if not all_data_frames:
        raise ValueError("No objects to concatenate: Parsing failed to extract any job data. Check if file is correctly formatted.")
        
    df_combined = pd.concat(all_data_frames, ignore_index=True)
    df_combined.columns = [col.strip('_') for col in df_combined.columns]
    return df_combined

# --- NEW FUNCTION FOR DATA AUGMENTATION ---
def augment_data_with_demographics(filename="synthetic_20_job_binary_dataset.csv"):
    """
    Loads the generated data and adds synthetic Age and CGPA features.
    This is required before training the recommender model.
    """
    try:
        df = pd.read_csv(filename)
    except FileNotFoundError:
        print(f"Error: {filename} not found for augmentation.")
        return

    num_rows = len(df)
    
    # 1. Generate Age (Ages 22 to 40, centered around 28)
    ages = np.random.normal(loc=28, scale=3, size=num_rows).round().astype(int)
    ages = np.clip(ages, 22, 40) 
    
    # 2. Generate CGPA (CGPA 6.0 to 9.8, centered around 7.8)
    cgpas = np.random.normal(loc=7.8, scale=1.0, size=num_rows)
    cgpas = np.clip(cgpas, 6.0, 9.8).round(1) 
    
    df['Age'] = ages
    df['CGPA'] = cgpas
    
    # Save the augmented file, overwriting the original synthetic file
    df.to_csv(filename, index=False)
    print(f"✅ Data augmentation complete. Added 'Age' and 'CGPA' columns to {filename}.")


# --- Execution Block (Uses Relative Path for XLSX) ---
if __name__ == "__main__":
    output_filename = "synthetic_20_job_binary_dataset.csv"
    
    # Ensure this matches the name of your Excel file in the MINOR folder
    input_file_name = "minor spread.xlsx"
    
    print(f"Attempting to read file from local directory: {input_file_name}")

    try:
        df_generated = parse_and_generate_data(
            input_file=input_file_name, 
            num_rows_per_job=500
        )

        df_generated.to_csv(output_filename, index=False)
        print(f"\n✅ Success! Data generation complete. File saved as: {output_filename}")
        
        # --- Augment the data immediately after saving ---
        augment_data_with_demographics(output_filename) 

    except FileNotFoundError:
        print("\n❌ ERROR: File Not Found.")
        print(f"The input file must be named '{input_file_name}' and be in the same folder as the Python script.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")