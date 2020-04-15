# Import OpenCV2 for image processing
import cv2

# Import numpy for matrices calculations
import numpy as np

import RPi.GPIO as GPIO    # Import Raspberry Pi GPIO library
from time import sleep# Import the sleep function from the time module



GPIO.setwarnings(False)    # Ignore warning for now
GPIO.setmode(GPIO.BOARD)   # Use physical pin numbering
GPIO.setup(12, GPIO.OUT, initial=GPIO.LOW)   # Set pin 8 to be an output pin and set initial value to low (off)


def led_on():
    GPIO.output(12, GPIO.HIGH)
    
    
def led_off():
    GPIO.output(12, GPIO.LOW)
# Create Local Binary Patterns Histograms for face recognization
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Load the trained mode
recognizer.read('./trainer.yml')

# Load prebuilt model for Frontal Face
cascadePath = "haarcascade_frontalface_default.xml"


# Create classifier from prebuilt model
faceCascade = cv2.CascadeClassifier(cascadePath);
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# Set the font style
font = cv2.FONT_HERSHEY_SIMPLEX

# Initialize and start the video frame capture
cam = cv2.VideoCapture(0)

# Loop
while True:
    # Read the video frame
    ret, im =cam.read()

    # Convert the captured frame into grayscale
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)

    # Get all face from the video frame
    faces = faceCascade.detectMultiScale(gray, 1.2,5)

    # For each face in faces
    for(x,y,w,h) in faces:

        # Create rectangle around the face
        cv2.rectangle(im, (x-20,y-20), (x+w+20,y+h+20), (0,255,0), 4)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = im[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        #for (ex,ey,ew,eh) in eyes:
            #cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(255,0,0),2)

        # Recognize the face belongs to which ID
        Id,conf = recognizer.predict(gray[y:y+h,x:x+w])
        #print(Id)
        print(conf)

        # Check the ID if exist 
        if((Id == 1) and (conf < 80)):
            Id = "Srini"
            led_on()
            
        elif((Id == 2) and (conf < 80)):
            Id = "Dinesh"
            led_on()

        elif((Id == 3) and (conf < 80)):
            Id = "Vignesh"
            led_on()
            
        elif((Id == 4) and (conf < 80)):
            Id = "Dhina"
            led_on()
            
        #If not exist, then it is Unknown
        else:
            Id = "Unknown" 
            led_off()

        # Put text describe who is in the picture
        cv2.rectangle(im, (x-22,y-90), (x+w+22, y-22), (0,255,0), -1)
        cv2.putText(im, str(Id), (x,y-40), font, 2, (255,255,255), 3)

    # Display the video frame with the bounded rectangle
    cv2.imshow('im',im) 

    # If 'q' is pressed, close program
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Stop the camera
cam.release()

# Close all windows
cv2.destroyAllWindows()
