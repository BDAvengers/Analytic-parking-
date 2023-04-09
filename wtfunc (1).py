# Power by Big Data Avengers
import torch
import torchvision
from torchvision import transforms as T
from PIL import Image
import cv2
import time
from threading import Thread
import queue

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()
# За ранее обученный модель faster rcnn

coco_names = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "street sign", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse",
              "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "hat", "backpack", "umbrella", "shoe",
              "eye glasses", "handbag", "tie", "suitcase",
              "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
              "plate", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
              "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog",
              "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed",
              "mirror", "dining table", "window", "desk", "toilet", "door", "tv",
              "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
              "oven", "toaster", "sink", "refrigerator", "blender", "book",
              "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush", "hair brush"]
# Объекты которые распознаются

frame_queue = queue.Queue(maxsize=2) # Память в очереди. То есть тут все работает очередно и приоритет стоит у главной функций.
def main(): # Главная функция входа в модуль. 
    cap = cv2.VideoCapture(0)

    while True:
        _, frame = cap.read()

        cv2.imshow('Video Stream', frame)
        if not frame_queue.full():
            frame_queue.put(frame)

        # задержка
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def func2(): # В определенный момент через try except, эта функция берет кадр из очереди и вызывает функцию обработки (car_detection)
    i = 0
    cars = 0
    while True:
        try:
            frame = frame_queue.get()
        except queue.Empty:
            continue

        image, carr = car_detection(frame)

        if carr:
            cars += 1
            cv2.imwrite(str(i) + '.jpg', image)
            i += 1

        	# распознование модели машины и разделение его по классам для дальнейшией аналитики клиентской базы, это то мы хотели добавить после этого

        print(cars)
        time.sleep(1)


def car_detection(frame): # В этой функций обрабатывается уже полученный кадр. То есть распознается машина.
    # ig = frame
    transform = T.ToTensor()
    img = transform(frame)

    with torch.no_grad():
        pred = model([img])

    # bboxes, labels, scores = pred[0]['boxes'], pred[0]['labels'], pred[0]['scores']
    # num = torch.argwhere(scores > 0.90).shape[0]

    car_bboxes = []
    car_labels = []
    car_scores = []
    font = cv2.FONT_HERSHEY_SIMPLEX

    i = 0
    # j = 0
    for ii in pred[0]['labels']:
        if ii == 3:
            car_bboxes.append(pred[0]['boxes'][i])
            car_labels.append(pred[0]['labels'][i])
            car_scores.append(pred[0]['scores'][i])
        i += 1

    num = 0
    for i in range(len(car_scores)):
        if car_scores[i] > 0.9:
            num += 1

    # igg = cv2.imread(frame)
    # igg = frame.copy()
    for i in range(num):
        x1, y1, x2, y2 = car_bboxes[i].numpy().astype('int')
        # class_name = coco_names[2]
        frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
        frame = cv2.putText(frame, "CAR", (x1, y1 - 10), font, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

    # objectCar = torch.argwhere(labels == 3).shape[0]
    if num != 0:
        carr = True
    else:
        carr = False

    return frame, carr


if __name__ == "__main__": # Использовали потоки чтобы реализировать очередь.
    # t1 = threading.Thread(target=main)
    # t2 = threading.Thread(target=func2)
    t1 = Thread(target=main)
    t2 = Thread(target=func2)
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    # thread1 = Thread(target=main)
    # thread1.start()
