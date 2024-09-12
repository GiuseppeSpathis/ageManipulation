from PIL import Image
import numpy as np

import argparse
import os
import subprocess

import pandas as pd


def predict_age(image_path):
    img = Image.open(image_path)
    img = np.array(img)

    command = "python ~/ageManipulation/pytorch-DEX/demo.py " + image_path 
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    
    stdout, _ = process.communicate()
    stdout = stdout.decode('utf-8')  # Convert bytes to string
    lines = stdout.split('\r\n')  # Split the output by lines
    for line in lines:
        if 'age:' in line:  # Find the line with the age
            age_str = line.split('age: ')[1]  # Split the line by 'age: '
            try:
                age = int(float(age_str[0] + age_str[1]))  # Try to convert the age to a float and then to an int
                return age
            except ValueError:
                pass

    
    
    

    
parser = argparse.ArgumentParser(description='caricare i risultati su excell')
parser.add_argument('folder_path', type=str)
parser.add_argument('--excel_path', type=str, default='risultati_DEX_Output2.xlsx', help='Percorso del file Excel di output')
args = parser.parse_args()

img_gen = []
age = []
pred_age = []
error = []

for filename in os.listdir(args.folder_path):
    if filename.endswith(".jpg") or filename.endswith(".png"): 
        image_path = os.path.join(args.folder_path, filename)
        parts = filename.split('_')
        actual_age = int(parts[1].split('.')[0])
        p_age = predict_age(image_path)
        img_gen.append(parts[0])
        age.append(actual_age)
        pred_age.append(p_age)
        if not np.isnan(p_age):
            error.append(abs(actual_age - p_age))
        else:
            error.append(np.nan) 
   
        
average_error = sum(error) / len(error)



df = pd.DataFrame({'img_gen': img_gen, 'eta': age, 'pred_age': pred_age, 'error': error})


# Calcolare le medie degli errori per fasce di età
age_ranges = [(10, 19), (20, 29), (30, 39), (40, 49), (50, 59), (60, 69), (70, 79), (80, 89), (90, 99)]
average_errors_by_range = {}

for age_range in age_ranges:
    start, end = age_range
    range_errors = df[(df['eta'] >= start) & (df['eta'] <= end)]['error']
    average_errors_by_range[f'{start}-{end}'] = np.nanmean(range_errors)

# Aggiungere la media degli errori per fasce di età al DataFrame
for idx, (age_range, avg_error) in enumerate(average_errors_by_range.items(), start=1):
    df.loc[idx, 'age_range'] = age_range
    df.loc[idx, 'average_error'] = avg_error

# Aggiungere la media generale degli errori
df.loc[0, 'age_range'] = 'Overall'
df.loc[0, 'average_error'] = average_error

# Salvare il DataFrame in un file Excel
df.to_excel(args.excel_path, index=False)

print(f"Risultati salvati in {args.excel_path}")
