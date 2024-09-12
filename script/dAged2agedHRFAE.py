import os
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import pairwise_distances

def load_image(file_path):
    return np.load(file_path)

def calculate_cosine_distance(image1, image2):
    return pairwise_distances(image1.reshape(1, -1), image2.reshape(1, -1), metric='cosine', n_jobs=-1)[0][0]

def main(dataset_dir, output_dir, result_file):
    global counter
    counter = 0
    # Lista degli intervalli di età
    age_ranges = [(10, 19), (20, 29), (30, 39), (40, 49), (50, 59), (60, 69), (70, 79), (80, 89), (90, 99)]

    # Lista per memorizzare i risultati
    results = []

    # Scorrere tutte le immagini nel dataset
    for file_name in os.listdir(dataset_dir):
        if file_name.endswith('.npy'):
            base_name = file_name[:-4]  # Rimuovi l'estensione .npy

            distances = {}
            for i in range(len(age_ranges)):
                if age_ranges[i][0] == 90:
                    break
                
                for age in range(age_ranges[i][0], age_ranges[i][1] + 1): 
                     aged_image_path = os.path.join(output_dir, f"{base_name}_age_{age}.npy")
  
                     if os.path.exists(aged_image_path):           
                         found_next_image = False
                         for next_age in range(age_ranges[i+1][0], age_ranges[i+1][1] + 1):
                       
                             next_aged_image_path = os.path.join(output_dir, f"{base_name}_age_{next_age}.npy")
                             if os.path.exists(next_aged_image_path):
                                 aged_image = load_image(aged_image_path)
                                 next_aged_image = load_image(next_aged_image_path)
                                 distance = calculate_cosine_distance(aged_image, next_aged_image)
                                 distances[f"{age_ranges[i]}-{age_ranges[i+1]}"] = distance
                                 found_next_image = True
#                         if found_next_image == False:
#                             distances[f"{age_ranges[i]}-{age_ranges[i+1]}"] = 0
#                     else:
#                         distances[f"{age_ranges[i]}-{age_ranges[i+1]}"] = 0

            distances["image"] = base_name
            results.append(distances)
            counter = counter + 1
            print(counter)
               
               

    # Creare un DataFrame con i risultati
    df = pd.DataFrame(results)

    # Salvare i risultati in un file Excel
    df.to_excel(result_file, index=False)

if __name__ == "__main__":
    dataset_dir = "/dgx_scratch/userexternal/gspathis/my_datasetAlign"
    output_dir = "/dgx_scratch/userexternal/gspathis/outputHRFAEAlign"
    result_file = "distancesAged2agedHRFAE_adaface.xlsx"
    main(dataset_dir, output_dir, result_file)

