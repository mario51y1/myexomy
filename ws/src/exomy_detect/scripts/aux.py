import os

# Paso 5: Definir la función para detectar personas y gatos en un video
import cv2
import numpy as np
video_path = './content/Input.mkv'
output_video_path = './content/output_video.mp4' 

def detect_people_cats(video_path):
    net = cv2.dnn.readNetFromDarknet('./darknet/yolov3.cfg', './darknet/yolov3.weights')
    layers_names = net.getLayerNames()
    output_layers = [layers_names[i - 1] for i in net.getUnconnectedOutLayers()]
    with open('./darknet/coco.names', 'r') as f:
      classes = [line.strip() for line in f.readlines()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    video = cv2.VideoCapture(video_path)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec de video para el archivo de salida
    output_video = cv2.VideoWriter(output_video_path, fourcc, 30.0, (width, height))
    frames = []  # Lista para almacenar los fotogramas procesados
    i = 0
    while True:
        print('in')
        i+=1
        ret, frame = video.read() 
        if not ret or i>10:
            print('breaking')
            break

        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (608, 608), swapRB=True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        class_ids = []
        confidences = []
        boxes = []
        info = ""

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5 and (class_id == 0 or class_id == 15):  # Clase 0: persona, Clase 15: gato
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])

        if len(boxes) > 0:
            for i in range(len(boxes)):
                if class_ids[i] == 0:  # Persona
                    info += "Persona: "
                elif class_ids[i] == 15:  # Gato
                    info += "Gato: "
                
                x, y, w, h = boxes[i]
                info += f"({x}, {y}, {w}, {h}) "

        if info == "":
            info = "Nada"

        frames.append((frame, info))  # Agregar el fotograma y la información a la lista

        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        for i in indices:
            
            box = boxes[i]
            x, y, w, h = box

            # Dibujar el contorno y etiqueta del objeto detectado
            label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
            color = colors[class_ids[i]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        frames.append(frame)  # Agregar el fotograma a la lista

        if output_video is None:
            output_video = cv2.VideoWriter(output_video_path, fourcc, 30.0, (width, height))

        output_video.write(frame)  

    video.release()
    output_video.release()
    print('out?')

    return frames

frames = detect_people_cats(video_path)
