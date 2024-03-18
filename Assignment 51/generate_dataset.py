import os
import pandas as pd
from deepface import DeepFace

def generate_dataset(path="images"):
    data_frame = pd.DataFrame()
    label = 0

    for folder in os.listdir(path):
        for img in os.listdir(f"{path}/{folder}"):
            try:
                embedding_objs = DeepFace.represent(img_path = f"{path}/{folder}/{img}", model_name="ArcFace")
            except:
                pass
            df = pd.DataFrame([embedding_objs[0]["embedding"]])
            df["Celeb_Label"] = label
            df["Celeb_name"] = folder
            data_frame = data_frame._append(df, ignore_index = True)

        label += 1
    data_frame.to_csv("dataset.csv")