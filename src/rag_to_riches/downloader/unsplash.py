# Taken almost verbatim from the above-mentioned resource.
from PIL import Image
import glob
import torch
import pickle
import zipfile
from IPython.display import display
from IPython.display import Image as IPImage
import os
from rag_to_riches import config
from tqdm.autonotebook import tqdm
from sentence_transformers import util
torch.set_num_threads(4)

data_dir =  "/Users/asifqamar/github/rag_to_riches/data" #config["paths"]["data"]

img_folder = data_dir + "/photos/"
if not os.path.exists(img_folder) or len(os.listdir(img_folder)) == 0:
    os.makedirs(img_folder, exist_ok=True)
    
    print("Path does not exist! Creating it...")
    
    zip_filename = 'unsplash-25k-photos.zip'
    photo_filename = f'{data_dir}/{zip_filename}'
    if not os.path.exists(photo_filename):   #Download dataset if does not exist
        util.http_get('http://sbert.net/datasets/'+zip_filename, photo_filename)
        
    #Extract all images
    with zipfile.ZipFile(photo_filename, 'r') as zf:
        for member in tqdm(zf.infolist(), desc='Extracting'):
            zf.extract(member, img_folder)
        