# ENCM509-Project

The goal of this project is to detect and identify faces using the open CV module on a live webcam feed.

## Instructions
To run our facial recognition software you will need to provide it with training data or have a .yml file called face_recognizer.yml that contains the training information for the model.

To run with new training data:
1. Create a folder inside training_data/ and title it the name you wish to be identified when it shows on screen e.g Jenna
2. Fill that folder with images of yourself, the algoirthm will work best if you use the same camera for the training images and running the model. We used the camera app on windows for ease of use.
3. If a face_recognizer.yml file exsits delete it or else the model will not train on your new data
4. Run python multi_recognition.py, this will train the model then start your web cam
5. The web cam view will display any faces it sees as well as who they are and the confidence level, if it is unsure of a face it will say "unknown"

To run with a pretrained model:
1. Run python multi_recognition.py, this will load the yml for the model then start your web cam
2. The web cam view will display any faces it sees as well as who they are and the confidence level, if it is unsure of a face it will say "unknown"

If you wish to log your confidence to a csv for any analysis use recognition_logger.py, it has the same functionality as multi_recognition.py but will output data into a csv. Create a results directory in the same directory as recognition_logger.py to store the CSV's.

## Analysis Notebooks
The analysis notebooks were not designed for another party to use so they have some hard coded titles and paths. Feel free to use them if you want but you will need to edit them depending on your csv path and the titles of the graphs.