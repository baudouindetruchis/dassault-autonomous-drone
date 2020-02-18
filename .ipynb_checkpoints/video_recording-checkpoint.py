#from VideoCapture import Device
import cv2

# Connect to fpv stream
#cam = Device()

cv2.VideoCapture(1)

frame_count = 0

while True:
#	cam.saveSnapshot('temp.jpg', quality=75)
#    frame = cam.getImage()
    frame_count += 1


    # Display the frame
    cv2.imshow('frame', frame)

    # Check key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
