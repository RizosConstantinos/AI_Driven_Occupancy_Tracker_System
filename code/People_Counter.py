# IK IV RIZOS KONSTANTINOS : Program that automatically counts people inside an area
# First we import the necessary packages
from pyimagesearch.centroidtracker import CentroidTracker
from pyimagesearch.trackableobject import TrackableObject
from imutils.video import VideoStream
from imutils.video import FPS
from multiprocessing import Manager
from multiprocessing import Process
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import os
import sys

# Insert the list of class labels that MobileNet SSD was trained to detect
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]

# Next we construct the argument parse and parse the arguments for each camera
args = {"prototxt" : "mobilenet_ssd/MobileNetSSD_deploy.prototxt", "model" : "mobilenet_ssd/MobileNetSSD_deploy.caffemodel"
        ,"prototxt1" : "mobilenet_ssd1/MobileNetSSD_deploy.prototxt", "model1" : "mobilenet_ssd1/MobileNetSSD_deploy.caffemodel",
        "confidence" : 0.4 , "skip-frames" : 30 }

# We set Grand_total equal to zero 0    
Grand_total=0    
    
# load our serialized model from disk two times , for each camera 
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
net1 = cv2.dnn.readNetFromCaffe(args["prototxt1"], args["model1"])

# open camera 1 and camera 2
print("[INFO] starting Entrance Door Camera ...")
vs = VideoStream(src=0).start()
time.sleep(2.0)
print("[INFO] starting Exit Door Camera...")
vs1 = VideoStream(src=2).start()
time.sleep(2.0)    

# initialize the frame dimensions
W = None
H = None

# instantiate our centroid tracker, then initialize a list to store
# each of our dlib correlation trackers, followed by a dictionary to
# map each unique object ID to a TrackableObject
ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
trackers = []
trackableObjects = {}

ct1 = CentroidTracker(maxDisappeared=40, maxDistance=50)
trackers1 = []
trackableObjects1 = {}


# initialize the total number of frames processed thus far, along
# with the total number of objects that have moved either up or down
totalFrames = 0
totalDown = 0
totalUp = 0
totalDown1 = 0
totalUp1 = 0

# start the frames per second throughput estimator
fps = FPS().start()

# loop over frames from the video stream
while True:
            # grab the next frame and handle if we are reading from either
            # VideoCapture or VideoStream
        frame = vs.read()
        frame1 = vs1.read()

            # resize the frame to have a maximum width of 700 pixels
            # then convert
            # the frame from BGR to RGB for dlib
        frame = imutils.resize(frame, width=700)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        frame1 = imutils.resize(frame1, width=700)
        rgb1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)

            
            # if the frame dimensions are empty, set them
        if W is None or H is None:
                (H, W) = frame.shape[:2]
        if W is None or H is None:
                (H, W) = frame1.shape[:2]

            # initialize the current status along with our list of bounding
            # box rectangles returned by either (1) our object detector or
            # (2) the correlation trackers
        status = "Waiting"
        rects = []
        status1 = "Waiting"
        rects1 = []

                
            # check to see if we should run a more computationally expensive
            # object detection method to aid our tracker
        if totalFrames % 30 == 0:
                    # set the status and initialize our new set of object trackers
                status = "Detecting"
                trackers = []
                status1 = "Detecting"
                trackers1 = []

                    # convert the frame to a blob and pass the blob through the
                    # network and obtain the detections
                blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
                net.setInput(blob)
                detections = net.forward()
                
                blob1 = cv2.dnn.blobFromImage(frame1, 0.007843, (W, H), 127.5)
                net1.setInput(blob1)
                detections1 = net1.forward()

                    # loop over the detections
                for i in np.arange(0, detections.shape[2]):
                            # extract the confidence (probability) associated
                            # with the prediction
                        confidence = detections[0, 0, i, 2]

                            # filter out weak detections by requiring a minimum
                            # confidence
                        if confidence > 0.4 :
                                    # extract the index of the class label from the
                                    # detections list
                                idx = int(detections[0, 0, i, 1])

                                    # if the class label is not a person, ignore it
                                if CLASSES[idx] != "person":
                                        continue

                                    # compute the (x, y)-coordinates of the bounding box
                                    # for the object
                                box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                                (startX, startY, endX, endY) = box.astype("int")

                                    # construct a dlib rectangle object from the bounding
                                    # box coordinates and then start the dlib correlation
                                    # tracker
                                tracker = dlib.correlation_tracker()
                                rect = dlib.rectangle(startX, startY, endX, endY)
                                tracker.start_track(rgb, rect)

                                    # add the tracker to our list of trackers so we can
                                    # utilize it during skip frames
                                trackers.append(tracker)
                                
                 # loop over the detections
                for i in np.arange(0, detections1.shape[2]):
                            # extract the confidence (probability) associated
                            # with the prediction
                        confidence1 = detections1[0, 0, i, 2]

                            # filter out weak detections by requiring a minimum
                            # confidence
                        if confidence1 > 0.4 :
                                    # extract the index of the class label from the
                                        # detections list
                                idx1 = int(detections1[0, 0, i, 1])
    
                                    # if the class label is not a person, ignore it
                                if CLASSES[idx1] != "person":
                                            continue

                                    # compute the (x, y)-coordinates of the bounding box
                                    # for the object
                                box1 = detections1[0, 0, i, 3:7] * np.array([W, H, W, H])
                                (startX, startY, endX, endY) = box1.astype("int")

                                    # construct a dlib rectangle object from the bounding
                                    # box coordinates and then start the dlib correlation
                                    # tracker
                                tracker1 = dlib.correlation_tracker()
                                rect1 = dlib.rectangle(startX, startY, endX, endY)
                                tracker1.start_track(rgb1, rect1)

                                    # add the tracker to our list of trackers so we can
                                    # utilize it during skip frames
                                trackers1.append(tracker1)                  
                                
                                
            # otherwise, we should utilize our object *trackers* rather than
            # object *detectors* to obtain a higher frame processing throughput
        else:
                    # loop over the trackers
                for tracker in trackers:
                            # set the status of our system to be 'tracking' rather
                            # than 'waiting' or 'detecting'
                        status = "Tracking"

                            # update the tracker and grab the updated position
                        tracker.update(rgb)
                        pos = tracker.get_position()

                            # unpack the position object
                        startX = int(pos.left())
                        startY = int(pos.top())
                        endX = int(pos.right())
                        endY = int(pos.bottom())

                            # add the bounding box coordinates to the rectangles list
                        rects.append((startX, startY, endX, endY))
                
                for tracker1 in trackers1:
                            # set the status of our system to be 'tracking' rather
                            # than 'waiting' or 'detecting'
                        status1 = "Tracking"

                            # update the tracker and grab the updated position
                        tracker1.update(rgb1)
                        pos1 = tracker1.get_position()

                            # unpack the position object
                        startX = int(pos1.left())
                        startY = int(pos1.top())
                        endX = int(pos1.right())
                        endY = int(pos1.bottom())

                            # add the bounding box coordinates to the rectangles list
                        rects1.append((startX, startY, endX, endY))
                                
                        
            # draw a horizontal line in the center of the frame -- once an
            # object crosses this line we will determine whether they were
            # moving 'up' or 'down'
        cv2.line(frame, (0, H // 2), (W, H // 2), (0, 0, 200), 2)
        cv2.line(frame1, (0, H // 2), (W, H // 2), (0, 0, 200), 2)
            # use the centroid tracker to associate the (1) old object
            # centroids with (2) the newly computed object centroids
        objects = ct.update(rects)
        objects1 = ct1.update(rects1)
            # loop over the tracked objects
        for (objectID, centroid) in objects.items():
                    # check to see if a trackable object exists for the current
                    # object ID
                to = trackableObjects.get(objectID, None)

                    # if there is no existing trackable object, create one
                if to is None:
                        to = TrackableObject(objectID, centroid)

                    # otherwise, there is a trackable object so we can utilize it
                    # to determine direction
                else:
                            # the difference between the y-coordinate of the *current*
                            # centroid and the mean of *previous* centroids will tell
                            # us in which direction the object is moving
                        y = [c[1] for c in to.centroids]
                        direction = centroid[1] - np.mean(y)
                        to.centroids.append(centroid)

                            # check to see if the object has been counted or not
                        if not to.counted:
                                    # if the direction is negative
                                    # AND the centroid is above the center
                                    # line, count the object
                                if direction < 0 and centroid[1] < H // 2:
                                        totalUp += 1
                                        to.counted = True

                                    # if the direction is positive 
                                    # AND the centroid is below the
                                    # center line, count the object
                                elif direction > 0 and centroid[1] > H // 2:
                                        totalDown += 1
                                        to.counted = True

                    # store the trackable object in our dictionary
                trackableObjects[objectID] = to

                    # draw both the ID of the object and the centroid of the
                    # object on the output frame
                text = "ID {}".format(objectID)
                cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
            
            # construct a tuple of information we will be displaying on the
            # frame
        
        for (objectID, centroid) in objects1.items():
                    # check to see if a trackable object exists for the current
                    # object ID
                to1 = trackableObjects1.get(objectID, None)

                    # if there is no existing trackable object, create one
                if to1 is None:
                        to1 = TrackableObject(objectID, centroid)

                    # otherwise, there is a trackable object so we can utilize it
                    # to determine direction
                else:
                            # the difference between the y-coordinate of the *current*
                            # centroid and the mean of *previous* centroids will tell
                            # us in which direction the object is moving 
                        y1 = [c[1] for c in to1.centroids]
                        direction1 = centroid[1] - np.mean(y1)
                        to1.centroids.append(centroid)

                            # check to see if the object has been counted or not
                        if not to1.counted:
                                    # if the direction is negative 
                                    # AND the centroid is above the center
                                    # line, count the object
                                if direction1 < 0 and centroid[1] < H // 2:
                                        totalUp1 += 1
                                        to1.counted = True

                                    # if the direction is positive 
                                    # AND the centroid is below the
                                    # center line, count the object
                                elif direction1 > 0 and centroid[1] > H // 2:
                                        totalDown1 += 1
                                        to1.counted = True

                    # store the trackable object in our dictionary
                trackableObjects1[objectID] = to1

                    # draw both the ID of the object and the centroid of the
                    # object on the output frame
                text = "ID {}".format(objectID)
                cv2.putText(frame1, text, (centroid[0] - 10, centroid[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.circle(frame1, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)    
        
        
        # Create the 'totalIn' and 'totalOut' which show us the total number of people going in or out
        # for each camera
        totalIn=totalDown-totalUp
        totalOut=totalDown1-totalUp1
        Grand_total=totalIn-totalOut

        # We set names on our data , which will be showing on the screen after launch
            
        info = [
                 ("Out", totalUp),
                 ("In", totalDown),
                 ("Status", status),
                 ("Total EntranceDoor", totalIn), ("Grand Total in the Area " , Grand_total)]
           

            # loop over the info tuples and draw them on our frame
        for (i, (k, v)) in enumerate(info):
                text = "{}: {}".format(k, v)
                cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 200), 2)
        
        
        info = [("In", totalUp1),
                ("Out", totalDown1),
                ("Status",status1),("Total ExitDoor",totalOut),
                ("Grand Total in the Area " , Grand_total)]
                            

            # loop over the info tuples and draw them on our frame
        for (i, (k, v)) in enumerate(info):
                text = "{}: {}".format(k, v)
                cv2.putText(frame1, text, (10, H - ((i * 20) + 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 200), 2)
                
                
            # show the output frame
        cv2.imshow("Entrance", frame)
        key = cv2.waitKey(1) & 0xFF
        cv2.imshow("ExitDoor", frame1)
        key = cv2.waitKey(1) & 0xFF
        key1 = cv2.waitKey(1) & 0xFF
            # if the `q` key was pressed, break from the loop
        if key == ord("q"):
                    break
        if key1 == ord("q"):
                    break        
            # increment the total number of frames processed thus far and
            # then update the FPS counter
        totalFrames += 1
        fps.update()
    # stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
    
    #stop the camera video stream
vs.stop()
    # close any open windows
cv2.destroyAllWindows()
    
    
    

 

    
    
    
    
    
