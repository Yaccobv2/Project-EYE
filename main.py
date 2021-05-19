import numpy as np
import cv2
import matplotlib as plt
import time
import threading



def save_frame(result,frame):
    cv2.imwrite('./database/detection' + "_time_" + str(result.tm_hour) + "_" + str(result.tm_min) + "_" + str(
        result.tm_sec) + '.jpg', frame)

CONFIDENCE_FILTER=0.5
THRESHOLD=0.3
SAVING_TIME=0.3

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
    blob = cv2.dnn.blobFromImage(frame, 1 / 255, (412, 412), swapRB=True, crop=False)
    yolo.setInput(blob)

    # make forward pass and calculate its time
    start = time.time()
    layerOutputs = yolo.forward(outputlayers)
    end = time.time()

    # print("time:"+str(end-start))


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
                result = time.gmtime()

                try:
                    saveThread = threading.Thread(target=save_frame, args=(result,frame))
                    saveThread.start()
                except:
                    print("error while saving")



    cv2.imshow("frame", frame)

    timetest.append(time.time()-testTimer)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print(sum(timetest)/len(timetest))
cap.release()
cv2.destroyAllWindows()
