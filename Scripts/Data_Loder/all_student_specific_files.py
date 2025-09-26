import os
import pandas as pd

def all_student_specific_files(base_path, file_types):
    """
    Load selected CSV files from each student's folder.
    
    Args:
        base_path (str): Root path where student folders are stored.
        file_types (list): List of file types like ['EYE', 'EGE'].

    Returns:
        pd.DataFrame: Combined DataFrame with Student_ID and File_Type columns.
    """
    all_data = []

    for student_id in range(1, 39):
        folder_path = os.path.join(base_path, str(student_id))

        for file_type in file_types:
            file_name = f"{student_id}_{file_type}.csv"
            file_path = os.path.join(folder_path, file_name)

            if os.path.exists(file_path):
                try:
                    df = pd.read_csv(file_path)
                    df['Student_ID'] = student_id
                    df['File_Type'] = file_type
                    all_data.append(df)
                    print(f"✅ Loaded: {file_path}")
                except Exception as e:
                    print(f"⚠️ Error reading {file_path}: {e}")
            else:
                print(f"❌ Missing: {file_path}")

    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)
        return final_df
    else:
        print("⚠️ No data loaded.")
        return pd.DataFrame()

#how to use

# sys.path.append('/content/drive/MyDrive/IITB/scripts')
# from load_student_csv_files import load_student_data_from_csvs

# base_path = '/content/drive/MyDrive/IITB/Data/processed/STData'
# file_types = ['EYE', 'EGE', 'RATINO']

# df = load_student_data_from_csvs(base_path, file_types)
# print(df.shape)
# df.head()
