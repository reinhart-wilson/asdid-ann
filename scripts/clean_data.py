
import os, sys

# Menambahkan path src ke Python path agar modul-modul dalam source code dapat dipanggil
working_dir = os.path.abspath(os.path.dirname(__file__))
src_dir = os.path.join(working_dir,'..','src')
sys.path.append(src_dir)

from dataprep.data_prep import DataPrep

# Inisiasi kelas DataPrep
dp = DataPrep()

# Lokasi data tambahan
input_folder = "../dataset/original_data"
output_folder = "../dataset/histogram_data"
os.listdir(input_folder)
for item in os.listdir(input_folder):
    item_path = os.path.join(input_folder, item)



    # Laksanakan penyiapan data jika item adalah sebuah folder
    if os.path.isdir(item_path):
        output_class_folder = os.path.join(output_folder, item)
        # dp.bulk_convert_images(item_path, output_class_folder)
        dp.resize_images(os.path.join(input_folder,item), output_class_folder, 224)
        
        # dp.crop_images(output_class_folder, output_class_folder, 224)
    else:
        print(f"Skipping {item}, nota a folder")
        
# dp.split_dataset(output_folder, output_folder+'_split', train_ratio=0.8, 
#                  val_ratio=0.1)