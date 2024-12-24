import ultralytics
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
# проверяем что доступно из оборудования
ultralytics.checks()

# Создаем модель. При первом вызове загружает веса, требуется интернет
model = YOLO("yolov8s.pt")

# Теперь попробуем обучить на собственном датасете
# он доступен по ссылке https://drive.google.com/file/d/1qS_yGj3vkmEuv9Fc3Xffqg6fE5C8m3AG/view?usp=drive_link

# перед обучением необходимо скорректировать пути в файле masked.yaml
# пути должны быть абсолютными
# обучение может занять много времени, особенно на CPU
# Обучение модели
model.train(data='D:\\Lab_DL_5\\masked.yaml', model="yolov8s.pt", epochs=1, imgsz=224, batch=16, 
            project='animals_classifier', val = True, verbose=True)

# Запускаем модель. На вход подаем изображения с диска, указав к нему путь
results = model("animals\\val\\images\\pixabay_wild_000838.jpg")
# Достаем результаты модели
result = results[0]
plt.imshow(result.plot())

# Запускаем модель. На вход подаем изображения с диска, указав к нему путь
results = model("animals\\val\\images\\pixabay_cat_002662.jpg")
# Достаем результаты модели
result = results[0]
plt.imshow(result.plot())

# Запускаем модель. На вход подаем изображения с диска, указав к нему путь
results = model("animals\\val\\images\\pixabay_dog_002239.jpg")
# Достаем результаты модели
result = results[0]
plt.imshow(result.plot())



# Запускаем модель. На вход подаем изображения с диска, указав к нему путь
results = model(".\\cat_dog.jpg")
# Достаем результаты модели
result = results[0]
plt.imshow(result.plot())

# Запускаем модель. На вход подаем изображения с диска, указав к нему путь
results = model(".\\wild_dog.jpg")
# Достаем результаты модели
result = results[0]
plt.imshow(result.plot())


# Запускаем модель. На вход подаем изображения с диска, указав к нему путь
results = model(".\\wild.jpg")
# Достаем результаты модели
result = results[0]
plt.imshow(result.plot())

# Даже исходное изображение
img = result.orig_img


# напишем свою функцию для отрисовки прямоугольников на изображении
def draw_bboxes(image, results):
    boxes = results[0].boxes.cpu()
    orig_h, orig_w = results[0].orig_shape # размеры изображения
    class_names = results[0].names             # названия классов  
    for box in boxes:    
        # достаем координаты, название класса и скор
        class_idx = box.cls
        confidence = box.conf
        
        # Будем отрисовывать только то в чем сеть хорошо уверена
        if confidence>0.7:
            x1, y1, x2, y2 = box.xyxy[0].numpy()
            # рисуем прямоугольник
            cv2.rectangle(
                image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2, cv2.LINE_AA
            )
            
            # подписываем название класса
            cv2.putText(
                image, class_names[class_idx.item()], (int(x1), int(y1-10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
            )
        
    return image

# вызовем функцию и выведем на экран то чот получилось
annotated_img = draw_bboxes(img, results)
plt.imshow(annotated_img)






