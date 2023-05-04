Computer Vision application written in Python, to detect empty car parking spots.
Uses YOLO algorithm for object detection of the vehicles, and opencv for drawing the bounding
boxes around the vehicles, and loading the video. 

To run the repository, simply clone the project, and then run the carCounter.py file.

How the project works: 
Parking spaces with cars in their lots will have red bounding boxes around them. As cars have larger non-zero quantities than empty parking lots, their bounding boxes will have higher pixel counts. 



https://user-images.githubusercontent.com/100171698/236207103-ccf95183-fb7f-4ae4-a4d2-5324bcc30ee7.mov

