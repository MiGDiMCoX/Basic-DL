# -*- coding: utf-8 -*-
import os
from glob import glob
import shutil

def create_yolo_structure(base_dir, class_names):
    for split in ['train', 'val']:
        split_dir = os.path.join(base_dir, split)
        images_dir = os.path.join(split_dir, 'images')
        labels_dir = os.path.join(split_dir, 'labels')
        
        # Создаем директории images и labels
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)
        
        for class_id, class_name in enumerate(class_names):
            class_dir = os.path.join(split_dir, class_name)
            image_paths = glob(os.path.join(class_dir, '*.jpg'))
            
            for image_path in image_paths:
                image_name = os.path.basename(image_path)
                label_name = os.path.splitext(image_name)[0] + '.txt'
                
                # Перемещаем изображение в images
                shutil.copy(image_path, os.path.join(images_dir, image_name))
                
                # Создаем метку в формате YOLO
                label_path = os.path.join(labels_dir, label_name)
                with open(label_path, 'w') as label_file:
                    # Вся картинка выделена под один класс
                    label_file.write(f"{class_id} 0.5 0.5 1.0 1.0\n")

# Папка с исходной структурой
base_dir = 'D:/Lab_DL_5/animals'
# Список классов с их названиями
class_names = ['cat', 'dog', 'wild']

# Создание структуры данных для YOLO
create_yolo_structure(base_dir, class_names)