import numpy as np
import cv2
import matplotlib as plt
import time
import threading
from datetime import datetime
import mysql.connector
from datetime import datetime
import os


# database
db= mysql.connector.connect(
    host="localhost",
    user="root",
    passwd="root",
    database="facemaskdb"

)

mycursor = db.cursor()

# system parameters
CONFIDENCE_FILTER=0.5
THRESHOLD=0.3
SAVING_TIME=4
inputFrameSize= (412,412)
dir='E:/GitHub/Project-EYE/database/'



# saving pictures to directory and logs to mysql datatable
def save_frame(frame,dir):
    date = str(datetime.now()).replace(":","_")[:-7]
    cv2.imwrite( dir + "detection" + "_time_" + date + '.jpg', frame)
    dir = str(dir + "detection" + "_time_" + date + '.jpg')
    mycursor.execute("INSERT INTO pictures(date, directory) VALUES (%s,%s)", (datetime.now(),dir))
    db.commit()

# clearing directory and datatable, created for testing purposes
def clearDB(dir):
    mycursor.execute("DELETE FROM pictures")
    mycursor.execute("ALTER TABLE pictures AUTO_INCREMENT = 1")
    db.commit()
    for picture in os.listdir(dir):
        print(picture+ "  deleted")
        os.remove(dir+picture)

# printing database logs, created for testing purposes
def printDB():
    mycursor.execute("SELECT * FROM pictures")
    for x in mycursor:
        print(x)




(W, H) = (None, None)

timetest=[]

# create instance of VideoCapture class
cap = cv2.VideoCapture(0)

# read neural network structure, weights and biases
yolo = cv2.dnn.readNetFromDarknet("./yolo-ANN/yolov4-tiny_custom_maski.cfg",
                                  "./yolo-ANN/yolov4-tiny_custom_best.weights")
outputlayers = yolo.getUnconnectedOutLayersNames()

# read labels
with open("./yolo-ANN/obj_maski.names", 'r') as f:
    LABELS = f.read().splitlines()

# create rgb colors for every label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")



savingTime = time.time()

while (True):

    # Capture frame-by-frame
    ret, frame = cap.read()

    #start saving timer
    testTimer=time.time()


    # if the frame dimensions are empty, grab them
    if W is None or H is None:
        (H, W) = frame.shape[:2]

    # create input matrix from frame, apply transformations and pass it to the first layer of ANN
    blob = cv2.dnn.blobFromImage(frame, 1 / 255, inputFrameSize, swapRB=True, crop=False)
    yolo.setInput(blob)

    # make forward pass and calculate its time
    start = time.time()
    layerOutputs = yolo.forward(outputlayers)
    end = time.time()




    # initialize our lists of detected bounding boxes, confidences and class IDs for every grabbed frame
    boxes = []
    confidences = []
    classIDs = []

    # use output of ANN
    for output in layerOutputs:
        for detection in output:

            # calculate highest score and get it`s confidence number
            score = detection[5:]
            class_id = np.argmax(score)
            confidence = score[class_id]

            # if confidence is higher than selected value of CONFIDENCE_FILTER create bounding box for every detection
            if confidence > CONFIDENCE_FILTER:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # get left corner coordinates of bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update our list of bounding box coordinates,
                # confidences, and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(class_id)

        # apply non-maxima suppression to overlapping bounding boxes with low confidence
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_FILTER,
                                THRESHOLD)

        # check if any bounding box exists
        if len(idxs) > 0:


            # plot bounding boxes
            for i in idxs.flatten():

                # get the bounding box coordinates
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])

                # draw a bounding box rectangle, label and confidence on the frame
                color = [int(c) for c in COLORS[classIDs[i]]]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                text = "{}: {:.4f}".format(LABELS[classIDs[i]],
                                           confidences[i])
                cv2.putText(frame, text, (x, y-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


            # saving frame
            if (time.time() - savingTime >= SAVING_TIME and classIDs.count(1)):

                print("saving")
                savingTime = time.time()

                try:
                    saveThread = threading.Thread(target=save_frame, args=(frame,dir))
                    saveThread.start()
                except:
                    print("error while saving")



    cv2.imshow("frame", frame)

    timetest.append(time.time()-testTimer)


    pressedKey = cv2.waitKey(1) & 0xFF

    if pressedKey & 0xFF == ord('q'):
        break

    if pressedKey & 0xFF == ord('s'):
        print("showing table")
        printDBThread = threading.Thread(target=printDB)
        printDBThread.start()

    if pressedKey & 0xFF == ord('c'):
        print("clearing table")
        clearDBThread = threading.Thread(target=clearDB, args=(dir,))
        clearDBThread.start()

print(sum(timetest)/len(timetest))
cap.release()
cv2.destroyAllWindows()
