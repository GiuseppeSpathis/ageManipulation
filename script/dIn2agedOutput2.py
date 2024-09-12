import os
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import pairwise_distances
import subprocess



def load_image(file_path):
    return np.load(file_path)

def calculate_cosine_distance(image1, image2):
    return pairwise_distances(image1.reshape(1, -1), image2.reshape(1, -1), metric='cosine', n_jobs=-1)[0][0]

def predict_age(image_path):
    command = f"python /dgx_scratch/userexternal/gspathis/pytorch-DEX/demo.py {image_path}"
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)

    age = 0

    stdout, _ = process.communicate()
    stdout = stdout.decode('utf-8')  # Convert bytes to string
    lines = stdout.split('\r\n')  # Split the output by lines
    for line in lines:
        if 'age:' in line:  # Find the line with the age
            age_str = line.split('age: ')[1]  # Split the line by 'age: '
            try:
                age = int(float(age_str.split()[0]))  # Convert the age to a float and then to an int
            except ValueError:
                pass
    print("eta predetta con successo")
    return age

def main(dataset_dir, output_dir, result_file):
    global counter
    counter = 0
    # Lista degli intervalli di età
    age_ranges = [(10, 19), (20, 29), (30, 39), (40, 49), (50, 59), (60, 69), (70, 79), (80, 89), (90, 99)]

    # Dizionario per memorizzare i risultati
    results = []

    # Scorrere tutte le immagini nel dataset
    for file_name in os.listdir(dataset_dir):
        if file_name.endswith('.npy'):
            base_name = file_name[:-4]  # Rimuovi l'estensione .npy
            original_image_path = os.path.join(dataset_dir, file_name)
            original_image = load_image(original_image_path)

            # Predire l'età usando PyTorch DEX
            image_png_path = os.path.join(dataset_dir, f"{base_name}.png")
            if os.path.exists(image_png_path):
                predicted_age = predict_age(image_png_path)
                predicted_age_range = None
                for age_range in age_ranges:
                    if age_range[0] <= predicted_age <= age_range[1]:
                        predicted_age_range = f"{age_range[0]}-{age_range[1]}"
                        break
            else:
                predicted_age_range = "N/A"

            distances = {}

            for age_range in age_ranges:
                found_image = False
                for age in range(age_range[0], age_range[1] + 1):
                    aged_image_path = os.path.join(output_dir, f"{base_name}_{age}.npy")
                    if os.path.exists(aged_image_path):
                        aged_image = load_image(aged_image_path)
                        print("calcolo distanza")
                        distance = calculate_cosine_distance(original_image, aged_image)
                        distances[f"{age_range[0]}-{age_range[1]}"] = distance
                        found_image = True
                        break

                if not found_image:
                    # Se non si trova un'immagine invecchiata per questo intervallo di età, usa l'immagine originale
                    distance = calculate_cosine_distance(original_image, original_image)
                    distances[f"{age_range[0]}-{age_range[1]}"] = distance

            distances['pred_age'] = predicted_age_range
            distances['image'] = base_name
            results.append(distances)
            print("appendo i risultati")
            counter = counter + 1
            print(counter)

    # Creare un DataFrame con i risultati
    df = pd.DataFrame(results)
 
    print("salvo su excel")
    # Salvare i risultati in un file Excel
    df.to_excel(result_file, index=False)

if __name__ == "__main__":
    dataset_dir = "/dgx_scratch/userexternal/gspathis/my_datasetAlign"
    output_dir = "/dgx_scratch/userexternal/gspathis/output2Align"
    result_file = "distancesIn2agedOutput2_adaface.xlsx"
    main(dataset_dir, output_dir, result_file)

