import shutil
import random
import os
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class DataPrep:
    
    def is_image(self, path):
        """
        Memeriksa apakah file pada path adalah cebuah citra

        Parameters
        ----------
        path : str
            Lokasi dari file.

        Returns
        -------
        bool
            True jika file adalah citra, False jika tidak.

        """
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
        """
        Membagi dataset ke dalam tiga subset: pelatihan, validasi, dan pengujian.
        Dataset harus terdiri dari subdirektori yang merepresentasikan setiap kelas,
        di mana setiap subdirektori berisi file gambar untuk kelas tersebut.
    
        Fungsi ini akan membuat folder `train`, `validation`, dan `test` di lokasi
        keluaran (`output_path`) dengan struktur yang sama seperti dataset asli.
        Gambar akan dipindahkan ke folder-folder tersebut berdasarkan rasio yang 
        diberikan.
    
        Parameters
        ----------
        dataset_path : str
            Jalur ke direktori dataset asli yang berisi subdirektori untuk 
            setiap kelas.
        output_path : str
            Jalur ke direktori keluaran tempat subset pelatihan, validasi, dan 
            pengujian akan disimpan.
        train_ratio : float, optional
            Rasio jumlah data yang digunakan untuk subset pelatihan. Nilai 
            default adalah 0.7.
        val_ratio : float, optional
            Rasio jumlah data yang digunakan untuk subset validasi. Nilai 
            default adalah 0.2.
        test_ratio : float, optional
            Rasio jumlah data yang digunakan untuk subset pengujian. Nilai 
            default adalah 0.1.
    
        Returns
        -------
        None
            Fungsi ini tidak mengembalikan nilai, tetapi menghasilkan struktur 
            folder dan file sesuai dengan subset pelatihan, validasi, dan 
            pengujian di lokasi keluaran.
    
        Notes
        -----
        - Rasio `train_ratio`, `val_ratio`, dan `test_ratio` harus berjumlah 1.0.
        """     
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
        Mengubah ukuran satu gambar sambil mempertahankan aspek rasio. 
        Ukuran output didasarkan pada sisi yang lebih kecil dari gambar asli, 
        di mana sisi lainnya akan disesuaikan secara proporsional.
    
        Parameters
        ----------
        input_path : str
            Jalur ke file gambar input yang akan diubah ukurannya.
        output_path : str
            Jalur untuk menyimpan file gambar yang telah diubah ukurannya.
        size : int
            Ukuran baru untuk sisi yang lebih kecil dari gambar. 
            Sisi lainnya akan disesuaikan untuk mempertahankan aspek rasio.
    
        Returns
        -------
        None
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
        Mengubah ukuran semua gambar dalam sebuah folder sambil mempertahankan 
        aspek rasio. Gambar hasil akan disimpan di folder keluaran dengan 
        struktur yang sama seperti folder input.
    
        Parameters
        ----------
        input_folder : str
            Jalur ke folder yang berisi gambar-gambar yang akan diubah ukurannya.
        output_folder : str
            Jalur ke folder tempat gambar hasil akan disimpan.
            Jika sama dengan `input_folder`, gambar hasil akan menimpa file asli.
        size : int
            Ukuran baru untuk sisi terpendek dari setiap gambar. 
            Sisi lainnya akan disesuaikan untuk mempertahankan aspek rasio.
    
        Returns
        -------
        None
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
            
    def bulk_convert_images(self, input_folder, output_folder, target_format='jpeg'):
        """
        Mengonversi semua gambar pada folder input ke format yang ditentukan.
        Tidak in-place.

        Parameters
        ----------
        input_folder : str
            Path folder yang berisi data input.
        output_folder : str
            Path folder tujuan data yang sudah dikonversi.
        target_format : str, optional
            Format citra yang diinginkan. Default: 'jpeg'.

        Returns
        -------
        None.
        """
        
        # Memastikan folder output tersedia, akan dibuat jika belum ada
        os.makedirs(output_folder, exist_ok=True)
        
        cur_idx = 0 # Nomor untuk penamaan file    
        for filename in os.listdir(input_folder):
            input_path = os.path.join(input_folder, filename)
    
            if os.path.isfile(input_path):
                try:
                    with Image.open(input_path) as img:
                        # Konversi gambar ke RGB jika target adalah JPG (PNG memiliki kanal RGBA)
                        if target_format.upper() == "JPEG":
                            img = img.convert("RGB")

                        output_filename = f"add_{os.path.basename(os.path.dirname(input_path))}_{cur_idx}.{target_format.lower()}"
                        output_path = os.path.join(output_folder, output_filename)
                        cur_idx += 1

                        img.save(output_path, target_format.upper())
    
                    print(f"Converted {filename} to {target_format.upper()}")
                except Exception as e:
                    print(f"Failed to convert {filename}: {e}")
            
            
    
    def delete_small_images(folder_path, min_size):
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
                    
                    
    
    def move_small_images(input_folder, output_folder, min_size):
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
    

            