import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from exomy_detect.msg import Detection
# init stuff
bridge = CvBridge()
net = cv2.dnn.readNetFromDarknet(
    '/home/mario51y1/Documents/VISION/darknet/yolov3.cfg', 
    '/home/mario51y1/Documents/VISION/darknet/yolov3.weights'
)
layers_names = net.getLayerNames()
output_layers = [layers_names[i - 1] for i in net.getUnconnectedOutLayers()]
with open('/home/mario51y1/Documents/VISION/darknet/coco.names', 'r') as f:
  classes = [line.strip() for line in f.readlines()]

colors = np.random.uniform(0, 255, size=(len(classes), 3))
width = 640
height = 480

pub_image = rospy.Publisher('annotated_image/image', Image, queue_size=10)
pub_data = rospy.Publisher('annotated_image/data', Detection, queue_size=10)
def detect(msg_image):
    image = bridge.imgmsg_to_cv2(msg_image, desired_encoding='passthrough')
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (608, 608), swapRB=True, crop=False)
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
    
    out = Detection()
    if len(boxes) > 0:
        cat_detected = False
        person_detected = False
        for i in range(len(boxes)):
            if class_ids[i] == 0 and not person_detected:  # Persona
                out.person_detected=True
                out.person_x=int(boxes[i][0])
                out.person_y=int(boxes[i][1])
                out.person_w=int(boxes[i][2])
                out.person_h=int(boxes[i][3])
                person_detected = True
                info += "Persona: "
            elif class_ids[i] == 15 and not cat_detected:  # Gato
                out.cat_detected=True
                out.cat_x=int(boxes[i][0])
                out.cat_y=int(boxes[i][1])
                out.cat_w=int(boxes[i][2])
                out.cat_h=int(boxes[i][3])
                cat_detected = True
                info += "Gato: "
            
            x, y, w, h = boxes[i]
            info += f"({x}, {y}, {w}, {h}) "

    if info == "":
        info = "Nada"

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    for i in indices:
        
        box = boxes[i]
        x, y, w, h = box

        # Dibujar el contorno y etiqueta del objeto detectado
        label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
        color = colors[class_ids[i]]
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # publish into ros stuff
    image_message = bridge.cv2_to_imgmsg(image, 'bgr8')
    pub_image.publish(image_message)
    pub_data.publish(out)


def main():
    rospy.init_node('image_detector')
    rospy.Subscriber('/pi_cam/image_raw', Image, detect)
    rospy.spin()
    pass

if __name__=='__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
