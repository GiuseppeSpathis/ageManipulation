from PIL import Image
import numpy as np

import argparse
import os
import subprocess
import shutil

def create_target_ages(num):
    lista = [18,28,38,48, 58, 68, 78, 88, 98,108]
    # Estraggo l'unità del numero in input
    unita_num = num % 10

    # Creo una lista vuota per contenere i numeri modificati
    lista_modificata = []

    # Modifico ogni elemento della lista
    for elemento in lista:
        # Estraggo le decine dell'elemento
        decine = elemento // 10 * 10

        # Sostituisco l'unità dell'elemento con l'unità del numero in input
        nuovo_elemento = decine + unita_num

        # Controllo che il nuovo elemento sia nel range 18-100 e non sia uguale al numero in input
        if 18 <= nuovo_elemento <= 100 and nuovo_elemento != num:
            # Aggiungo il nuovo elemento alla lista modificata
            lista_modificata.append(nuovo_elemento)

    return lista_modificata



def predict_age(image_path):
    img = Image.open(image_path)
    img = np.array(img)

    command = "python ./pytorch-DEX/demo.py " + image_path 
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    
    age = 0
    man = 0.0
    woman = 0.0
    
    stdout, _ = process.communicate()
    stdout = stdout.decode('utf-8')  # Convert bytes to string
    lines = stdout.split('\r\n')  # Split the output by lines
    for line in lines:
        if 'age:' in line:  # Find the line with the age
            age_str = line.split('age: ')[1]  # Split the line by 'age: '
            try:
                age = int(float(age_str))  # Try to convert the age to a float and then to an int
            except ValueError:
                pass
        if 'woman:' in line:  # Find the line with the woman
            woman_str = line.split('woman: ')[1]  # Split the line by 'woman_str: '
            try:
                woman = float(woman_str)  # Try to convert the woman_str to a float and then to an int
            except ValueError:
                pass
        if 'man:' in line:  # Find the line with the woman
            man_str = line.split('man: ')[1]  # Split the line by 'woman_str: '
            try:
                man = float(man_str)  # Try to convert the woman_str to a float and then to an int
            except ValueError:
                pass
    
    
    
    gender = "male"
    
    if(woman >= man):
        gender = "female"
    
    
    target_ages = create_target_ages(age)
    
    current_dir = os.getcwd()
 
 
    new_image_path = current_dir + args.output_dir.replace("./", os.sep)
    
   
    # Call the age_editing.py script with the appropriate arguments
    subprocess.run([
        "python", "age_editing.py",
        "--image_path", image_path,
        "--age_init", str(age),
        "--gender", gender,
        "--save_aged_dir", "./output",
        "--specialized_path", "runwayml/stable-diffusion-v1-5",
        "--target_ages", *map(str, target_ages)
    ])

    shutil.move(image_path, new_image_path)

    print(f"L'immagine è stata spostata da {image_path} a {new_image_path}")

parser = argparse.ArgumentParser(description='Prevede l\'età di una persona in un\'immagine.')
parser.add_argument('folder_path', type=str, help='Il percorso della cartella contenente le immagini.')
parser.add_argument('output_dir', type=str, help='Il percorso della cartella dove spostare le immagini.')
args = parser.parse_args()


for filename in os.listdir(args.folder_path):
    if filename.endswith(".jpg") or filename.endswith(".png"): 
        image_path = os.path.join(args.folder_path, filename)
        predict_age(image_path)
