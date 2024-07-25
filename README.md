# AttendanceApp

1) Select if you want to register in the database or give attendance.
2) If you select 'Register', you will have to fill a form with the Student Name and Student ID Number.
   If you enter a Student ID Number that already exists in the database, you will be notified (since Student ID is unique).
3) Once the student details are accepted, face registration occurs where 20 photos will be taken and stored in the Dataset.
   The VGG model will be trained on the faces extracted from these photos.
5) After model training is complete, user is routed back to the homepage.
6) If you select 'Attendence', then the webcam will take a single photo and give the predicted label (Student ID Number) as output.
   The Timestamp and Student ID Number will be saved to a csv file called 'Attendance Record'.


------

Process:

1) Data Preparation:
   - Dataset organized by candidate names.
   - Label assignments saved to candidate_labels.csv.

2) Model Training:
   - Images preprocessed and labeled.
   - Model trained and saved as fine_tuned_VGG16_model.h5.

3) Prediction:
   - Trained model and label map loaded.
   - New image processed and predicted.
   - Prediction translated into a candidate ID using candidate_labels.csv.
