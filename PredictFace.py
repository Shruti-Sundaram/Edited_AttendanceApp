import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
from pathlib import Path
from datetime import datetime
from tensorflow.keras.models import load_model
import os
import csv


# Function to preprocess the image
def preprocess_image(image_path, target_size=(224, 224)):
    img = cv2.imread(image_path)
    img = cv2.resize(img, target_size)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Function to predict the label of the image
def predict_candidate(image_path):

    model = load_model('fine_tuned_VGG16_model.h5')

    df = pd.read_csv('candidate_labels.csv')
    labels_dict = df.set_index('Label')['Student ID'].to_dict()

    if image_path == '':
        image_path = 'Dataset/test_img.jpg' 

    print("Image Path is", image_path)

    img = preprocess_image(image_path)
    prediction = model.predict(img)
    print("prediction:",prediction)
    predicted_class = np.argmax(prediction, axis=1)[0]
    print("predicted_class:",predicted_class)
    predicted_label = labels_dict[predicted_class]
    prob = prediction[0, predicted_class]
    if prob < 0.9:
        predicted_label = '000'


    student_id = ''
    student_name = ''
    student_class = '' 


    # Get the current timestamp in 12-hour format without seconds
    timestamp = datetime.now().strftime('%Y-%m-%d %I:%M %p')
    # Save to CSV
    if os.path.exists('student_record.csv'):
        with open('student_record.csv', mode='r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if str(row['student_id']) == str(predicted_label) and predicted_label != '000':
    
                    student_id = row['student_id']
                    student_name = row['student_name']
                    student_class = row['student_class']

                    attendance_record = pd.DataFrame([[timestamp, student_id,student_name,student_class]], columns=['Timestamp', 'Student ID Number','Student Name','Class Room'])
                    attendance_record.to_csv('Attendance_Record.csv', mode='a', header=False, index=False)
                                    
            
    

    print("Attendence marked for StudentID:", predicted_label)
    return predicted_label

