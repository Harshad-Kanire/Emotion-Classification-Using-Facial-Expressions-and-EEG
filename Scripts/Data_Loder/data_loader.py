import os
import pandas as pd

def load_all_students_all_files(base_path):
    all_data = []

    for student_id in range(1, 39):  # for students 1 to 38
        student_folder = os.path.join(base_path, str(student_id))
        if not os.path.exists(student_folder):
            print(f"❌ Folder missing: {student_folder}")
            continue

        for file_name in os.listdir(student_folder):
            if file_name.endswith('.csv'):
                file_path = os.path.join(student_folder, file_name)
                try:
                    df = pd.read_csv(file_path)
                    file_type = file_name.split('_')[1].replace('.csv', '')  # example: EYE
                    df['Student_ID'] = student_id
                    df['File_Type'] = file_type
                    all_data.append(df)
                    print(f"✅ Loaded: {file_path}")
                except Exception as e:
                    print(f"⚠️ Error reading {file_path}: {e}")

    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)
        return final_df
    else:
        print("⚠️ No data found.")
        return pd.DataFrame()

#how to use

# base_path = '/content/drive/MyDrive/IITB/Data/processed/STData'
# df_all = load_all_students_all_files(base_path)

# print(df_all.shape)
# df_all.head()
