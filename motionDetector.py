import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# Function for motion detection
def motionDetection():
    # Capture video from file
    cap = cv.VideoCapture("./img/vtest.avi")
    # Read the first frame
    ret, frame1 = cap.read()
    # Read the second frame
    ret, frame2 = cap.read()

    # Loop until video capture is open
    while cap.isOpened():
        # Compute the absolute difference between the two frames
        diff = cv.absdiff(frame1, frame2)
        # Convert the difference to grayscale
        diff_gray = cv.cvtColor(diff, cv.COLOR_BGR2GRAY)
        # Apply Gaussian blur to the grayscale difference
        blur = cv.GaussianBlur(diff_gray, (5, 5), 0)
        # Threshold the blurred difference
        _, thresh = cv.threshold(blur, 20, 255, cv.THRESH_BINARY)
        # Dilate the thresholded image to fill gaps
        dilated = cv.dilate(thresh, None, iterations=3)
        # Find contours in the dilated image
        contours, _ = cv.findContours(
            dilated, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        # Loop through each contour found
        for contour in contours:
            # Get the bounding rectangle of the contour
            (x, y, w, h) = cv.boundingRect(contour)
            # Ignore small contours (noise)
            if cv.contourArea(contour) < 900:
                continue
            # Draw a rectangle around the detected motion
            cv.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)
            # Add text to indicate motion status
            cv.putText(frame1, "Status: {}".format('Movement'), (10, 20), cv.FONT_HERSHEY_SIMPLEX,
                       1, (255, 0, 0), 3)

        # Show the resulting frame with motion detection
        cv.imshow("Video", frame1)
        # Set the current frame as the previous frame for the next iteration
        frame1 = frame2
        # Read the next frame
        ret, frame2 = cap.read()

        # Exit loop if 'Esc' key is pressed
        if cv.waitKey(50) == 27:
            break

    # Release video capture and close all windows
    cap.release()
    cv.destroyAllWindows()

# Main function
if __name__ == "__main__":
    # Call the motionDetection function
    motionDetection()
