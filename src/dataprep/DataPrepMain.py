import os
from data_prep import DataPrep



def get_folders_in_directory(directory):
    # Membuat list kosong untuk menyimpan nama-nama folder
    folders_list = []
    
    # Loop untuk setiap item di dalam direktori
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        
        # Cek apakah item merupakan folder dan bukan file
        if os.path.isdir(item_path):
            folders_list.append(item)
    
    return folders_list



if __name__ == "__main__":

    dataset_path = "..\Dataset\original_data"
    output_dir = "..\../Dataset/split_prepped_additional_data_2/test"
    split_dir = "..\..\Dataset\split_prepped_additional_data_2_2"
    scrapped_dir = "..\Dataset\scrapped_data"
    seed=42
    dpu = DataPrep(seed)
    
    # data_folder_names = get_folders_in_directory(dataset_path)
    
    new_size = 224  # Ukuran lebar yang diinginkan untuk gambar baru
    
    
    # for folder_name in data_folder_names:
    #     input_folder_path = os.path.join(dataset_path, folder_name)  # Path folder input
    #     output_folder_path = os.path.join(output_dir, folder_name)   # Path folder output
    #     scrapped_folder_path = os.path.join(scrapped_dir, folder_name)
        
    #     # Pindahkan gambar yang terlalu kecil
    #     dpu.move_small_images(input_folder_path, scrapped_folder_path, new_size)
        
    #     # Resize dan potong
    #     dpu.resize_images(input_folder_path, output_folder_path, new_size)
    #     dpu.crop_images(output_folder_path, output_folder_path, new_size)
        
    dpu.split_dataset(output_dir, split_dir, train_ratio=0, val_ratio=0.5, test_ratio=0.5)

    # bacterial_blight_path = os.path.join(split_dir,'train', 'bacterial_blight')
    # downey_mildew_path = os.path.join(split_dir,'train', 'downey_mildew')
    # augmented_dir = "..\Dataset\\augmented_data"
    # split_augmented_dir = "..\Dataset\\split_augmented_data"
    
    # dpu.generate_augmented_image(bacterial_blight_path, 
    #                              os.path.join(augmented_dir,'bacterial_blight'), 
    #                              2
    #                              )
    # dpu.generate_augmented_image(downey_mildew_path, 
    #                              os.path.join(augmented_dir,'downey_mildew'), 
    #                              2
    #                              )
        
    # dpu.split_dataset(augmented_dir, split_augmented_dir)
