import numpy as np
import cv2
from gtts import gTTS
import os

net=cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes=[]
with open('coco.names','r')as f:
    classes=f.read().splitlines()
# detection of objects using an image
# img=cv2.imread('study-room.jpg')

# detection of objects using webcam: Change 0 to any video for video capture rather than webcam capture
cap = cv2.VideoCapture(0)
a=0
# add while loop for video capture
while True:
    a = a + 1

    _, img= cap.read()
    height,width, _ = img.shape

    blob=cv2.dnn.blobFromImage(img,1/255,(416,416),(0,0,0),swapRB=True,crop=False)

    net.setInput(blob)
    outputLayers=net.getUnconnectedOutLayersNames()
    layerOutputs=net.forward(outputLayers)

    boxes=[]
    confidences=[]
    class_ids=[]

    for output in layerOutputs:
        for detection in output:
            scores=detection[5:]
            class_id=np.argmax(scores)
            confidence=scores[class_id]
            if confidence>0.5:
                center_x=int(detection[0]*width)
                center_y=int(detection[1]*height)
                w=int(detection[2]*width)
                h=int(detection[3]*height)

                x= int(center_x - w/2)
                y= int(center_y - h/2)

                boxes.append([x,y,w,h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)

    #print(len(boxes))
    indexes= cv2.dnn.NMSBoxes(boxes,confidences,0.5,0.4)
    #print(indexes.flatten())

    font=cv2.FONT_HERSHEY_PLAIN
    colors=np.random.uniform(0,255,size=(len(boxes),3))

    if len(indexes)> 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label= str(classes[class_ids[i]])
            print("Object Detected: ", label)
            color= colors[i]
            confidence= str(round(confidences[i],2))
            cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
            cv2.putText(img, label + " " + confidence,(x,y+20),font,2, (255,255,255), 2)
            language='en'
            output=gTTS(text=label,lang=language,slow=True)
            output.save("output.mp3")
            os.system("start output.mp3")


        cv2.imshow('Capturing...',img)
        #cv2.waitKey(0)
        #for video capture change the argument passed to waitkey to 1
        key=cv2.waitKey(1)
        if key == ord('q'):
            break

print("Total frames captured: ", a)
cap.release()
cv2.destroyAllWindows()