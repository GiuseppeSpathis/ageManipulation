from PIL import Image
import numpy as np
from deepface import DeepFace
import argparse
import os

import pandas as pd


def predict_age(image_path):
    img = Image.open(image_path)
    img = np.array(img)

    

    try:
        result = DeepFace.analyze(img, actions=['age'])
        return result[0]["age"]
    except ValueError:
        return np.nan

    # Generate target ages based on the predicted age
    
    
    
    
    

    
parser = argparse.ArgumentParser(description='caricare i risultati su excell')
parser.add_argument('folder_path', type=str)
parser.add_argument('--excel_path', type=str, default='risultati.xlsx', help='Percorso del file Excel di output')
args = parser.parse_args()

img_gen = []
age = []
pred_age = []
error = []

for filename in os.listdir(args.folder_path):
    if filename.endswith(".jpg") or filename.endswith(".png"): 
        image_path = os.path.join(args.folder_path, filename)
        parts = filename.split('_')
        img_gen.append(parts[0])
        age.append(parts[1].split('.')[0])
        pred_age.append(predict_age(image_path))
        error.append(abs(int(parts[1].split('.')[0]) - predict_age(image_path)))
        
average_error = np.nanmean(np.array(error))



df = pd.DataFrame({'img_gen': img_gen, 'eta': age, 'pred_age': pred_age, 'error': error})

df.loc[0, 'average'] = average_error

df.to_excel(args.excel_path, index=False)

print(f"Risultati salvati in {args.excel_path}")