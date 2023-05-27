## Paso 1: Clonar el repositorio de Darknet
git clone https://github.com/AlexeyAB/darknet.git
#!git clone https://github.com/LorenaPara/cat_detection
cd darknet

# Paso 2: Habilitar GPU (si estás usando Google Colab con GPU)
sed -i 's/OPENCV=0/OPENCV=1/' Makefile
sed -i 's/GPU=0/GPU=1/' Makefile
make

# Paso 3: Descargar los archivos de configuración y pesos pre-entrenados de YOLOv4
wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov3.weights
wget https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov3.cfg
wget https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names

