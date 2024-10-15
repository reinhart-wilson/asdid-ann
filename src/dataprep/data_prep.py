import shutil
import random
import os
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class DataPrep:
    
    def __init__(self, seed=None):
        if seed:        
            random.seed(seed)
            np.random.seed(seed)
            self.seed = seed
    
    def is_image(self, path):
        # Cek apakah file adalah gambar
        valid_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp',
                            '.jfif')
        if not path.lower().endswith(valid_extensions):
            return False
        if not os.path.isfile(path):
            return False
        return True    
    

    def split_dataset(self, dataset_path, output_path, train_ratio=0.7, 
                      val_ratio=0.2, test_ratio=0.1):
        
        # Buat folder-folder keluaran untuk menyimpan hasil split
        train_dir = os.path.join(output_path, 'train')
        val_dir = os.path.join(output_path, 'validation')
        test_dir = os.path.join(output_path, 'test')
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)
    
        # Iterasi untuk setiap folder kelas dalam dataset asli
        for class_folder in os.listdir(dataset_path):
            class_path = os.path.join(dataset_path, class_folder)
            if os.path.isdir(class_path):
                images = os.listdir(class_path)
                random.shuffle(images)
    
                # Hitung banyak data untuk tiap split
                num_images = len(images)
                num_train = int(train_ratio * num_images)
                num_val = int(val_ratio * num_images)
    
                # Split set data
                train_images = images[:num_train]
                val_images = images[num_train:num_train + num_val]
                test_images = images[num_train + num_val:]
    
                # Salin gambar ke folder keluaran
                for image in train_images:
                    src_path = os.path.join(class_path, image)
                    dest_path = os.path.join(train_dir, class_folder, image)
                    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                    shutil.copy(src_path, dest_path)
                for image in val_images:
                    src_path = os.path.join(class_path, image)
                    dest_path = os.path.join(val_dir, class_folder, image)
                    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                    shutil.copy(src_path, dest_path)
                for image in test_images:
                    src_path = os.path.join(class_path, image)
                    dest_path = os.path.join(test_dir, class_folder, image)
                    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                    shutil.copy(src_path, dest_path)
                    
                  
                    
    def resize_image(self, input_path, output_path, size):
        """
        Mengubah satu gambar dengan mempertahankan aspek rasio. Pengguna hanya
        bisa memilih salah satu ukuran yang akan menjadi patokan: lebar atau
        tinggi. Secara default, width akan menjadi patokan sehingga tinggi baru 
        gambar akan secara otomatis disesuaikan dengan memperhitungkan aspek 
        rasio. Jika pengguna ingin mengatur tinggi, maka argumen opsional 
        height harus diisi.
    
        Args
        ----------
        input_path : str 
            Path gambar input.
        output_path : str
            Path gambar output.
        width : int
            Lebar baru dari gambar.
        height : int, opsional
            DESCRIPTION. Secara default bernilai None.

        """
        # Cek apakah file adalah gambar
        if not self.is_image(input_path):
            print(f"Skipping {input_path}, not an image file.")
            return
        
        try:
            with Image.open(input_path) as img:
                width, height = img.size
                
                if height < width:
                    new_height = size
                    new_width = int((width / height) * size)
                else:
                    new_width = size
                    new_height = int((height / width) * size)
                resized_img = img.resize((new_width, new_height), Image.LANCZOS)
                resized_img.save(output_path)
        except Exception as e:
            print(f"Error processing {input_path}: {e}")
            
            
                 
    def resize_images(self, input_folder, output_folder, size):
        """
        Mengubah ukuran semua gambar yang disimpan di suatu direktori. Perlu 
        diperhatikan bahwa parameter size adalah lebar baru dari gambar. Tinggi
        baru gambar akan secara otomatis disesuaikan dengan memperhitungkan 
        aspek rasio.
    
        Args
        ----------
        input_folder : str
            Path folder penyimpanan gambar-gambar yang akan diubah ukurannya.
        output_folder : str
            Path folder tempat gambar-gambar baru disimpan. Jika diisi dengan 
            None, maka fungsi ini akan menimpa gambar di folder input.
        size : int
            Lebar baru dari gambar. 
        """
        if output_folder==None:
            output_folder=input_folder
        else:            
            # Pastikan folder output ada atau buat jika belum ada
            if not os.path.exists(output_folder):
                print(output_folder)
                os.makedirs(output_folder)
    
        # Loop untuk setiap file dalam folder input
        for filename in os.listdir(input_folder):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)    
            self.resize_image(input_path, output_path, size)
            
            
        
    def crop_image(self, input_path, output_path, x):
        """
        Memotong gambar sehingga bagian yang dihasilkan adalah bagian tengah 
        dengan ukuran x kali x.

        Parameters
        ----------
        image : TYPE
            DESCRIPTION.
        x : TYPE
            DESCRIPTION.

        Returns
        -------
        cropped_image : TYPE
            DESCRIPTION.

        """
        # Cek apakah file adalah gambar
        if not self.is_image(input_path):
            print(f"Skipping {input_path}, not an image file.")
            return   
        
        try:
            with Image.open(input_path) as img:
                # Tentukan koordinat untuk proses cropping
                width, height = img.size
                left = (width - x) / 2
                top = (height - x) / 2
                right = (width + x) / 2
                bottom = (height + x) / 2
                cropped_img = img.crop((left, top, right, bottom))
                
                cropped_img.save(output_path)
        except Exception as e:
            print(f"Error processing {input_path}: {e}")
            


    def crop_images(self, input_folder, output_folder, x=224):
        """
        Memotong semua gambar yang disimpan di suatu direktori. Bagian yang 
        dihasilkan adalah bagian tengah dengan ukuran x kali x.
    
        Args
        ----------
        input_folder : str
            Path folder penyimpanan gambar-gambar.
        output_folder : str
            Path folder tempat gambar-gambar baru disimpan. Jika diisi dengan 
            None, maka fungsi ini akan menimpa gambar di folder input.
        size : int
            Lebar baru dari gambar. 
        """
        if output_folder==None:
            output_folder=input_folder
        else:            
            # Pastikan folder output ada atau buat jika belum ada
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
    
        # Loop untuk setiap file dalam folder input
        for filename in os.listdir(input_folder):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)    
            self.crop_image(input_path, output_path, x)
            
            
    
    def delete_small_images(self,folder_path, min_size):
        # Loop melalui semua file dalam folder
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            
            # Pastikan hanya memproses file gambar
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                try:
                    with Image.open(file_path) as img:
                        width, height = img.size
                        
                        # Jika lebar atau tinggi lebih kecil dari min_size, hapus gambar
                        if width < min_size or height < min_size:
                            os.remove(file_path)
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
                    
                    
    
    def move_small_images(self, input_folder, output_folder, min_size):
        # Pastikan folder tujuan ada, jika tidak, buat folder tersebut
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        # Loop melalui semua file dalam folder sumber
        for filename in os.listdir(input_folder):
            source_path = os.path.join(input_folder, filename)
            
            # Pastikan hanya memproses file gambar
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                try:
                    with Image.open(source_path) as img:
                        width, height = img.size
                        
                        # Jika lebar atau tinggi lebih kecil dari min_size, pindahkan gambar
                        if width < min_size or height < min_size:
                            destination_path = os.path.join(output_folder, 
                                                            filename)
                            shutil.move(source_path, destination_path)
                except Exception as e:
                    print(f"Error processing {source_path}: {e}")
                    
    
    
    def generate_augmented_image(self, input_dir, augmented_dir, num_images):
        """
        

        Parameter
        ----------
        input_dir : TYPE
            DESCRIPTION.
        augmented_dir : TYPE
            DESCRIPTION.
        num_images : int
            Banyak gambar teraugmentasi dari 1 gambarnya.
        seed : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """
        # Buat direktori untuk menyimpan gambar yang telah di-augmentasi
        os.makedirs(augmented_dir, exist_ok=True)
        
        # ImageDataGenerator untuk augmentasi
        
        datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='constant', 
            cval=0)
        
        # Augmentasi gambar-gambar dalam kelas minoritas
        for image_name in os.listdir(input_dir):
            image_path = os.path.join(input_dir, image_name)
            image = Image.open(image_path)
            image_array = np.array(image) 
            image_array = image_array.reshape((1,) + image_array.shape) #mengubah bentuk image menjadi batch untuk menyesuaikan tipe input flow
        
            i = 0
            for batch in datagen.flow(image_array, batch_size=1,
                                      save_to_dir=augmented_dir,
                                      save_prefix='aug',
                                      save_format='jpeg',):
                i += 1
                if i >= num_images:  
                    break
    
    
    def bulk_convert_images(self, input_folder, output_folder, target_format='jpeg'):
        """
        Converts all images in the input folder to the specified format.
    
        Parameters:
        - input_folder (str): Path to the folder containing the input images.
        - output_folder (str): Path to the folder where converted images will be saved.
        - target_format (str): The desired format (e.g., "JPEG", "PNG").
        """
        
        # Ensure the output folder exists
        os.makedirs(output_folder, exist_ok=True)
    
        # Nomor untuk penamaan file
        cur_idx = 0
    
        # Loop through all the files in the input folder
        for filename in os.listdir(input_folder):
            input_path = os.path.join(input_folder, filename)
    
            # Check if the item is a file, not a folder
            if os.path.isfile(input_path):
                try:
                    # Open the image
                    with Image.open(input_path) as img:
                        # Convert the image to RGB mode if converting to JPEG (because JPEG doesn't support transparency)
                        if target_format.upper() == "JPEG":
                            img = img.convert("RGB")
                        
                        # Create the output path with the new extension
                        output_filename = f"add_{os.path.basename(os.path.dirname(input_path))}_{cur_idx}.{target_format.lower()}"
                        output_path = os.path.join(output_folder, output_filename)
                        cur_idx += 1
                        
                        # Save the image in the specified format
                        img.save(output_path, target_format.upper())
    
                    print(f"Converted {filename} to {target_format.upper()}")
                except Exception as e:
                    print(f"Failed to convert {filename}: {e}")
            