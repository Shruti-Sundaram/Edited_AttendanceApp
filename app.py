from flask import Flask, render_template, request, jsonify
import os
import base64
import numpy as np
import cv2
import tensorflow as tf
import csv
from ModelTransferLearning import ModelFineTuning
# from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import base64
from io import BytesIO
from PIL import Image
from PredictFace import preprocess_image, predict_candidate
import webbrowser
from threading import Timer


app = Flask(__name__)

# Load the Caffe model
net = cv2.dnn.readNetFromCaffe('models/deploy.prototxt', 'models/res10_300x300_ssd_iter_140000.caffemodel')

def open_browser():
      webbrowser.open_new('http://127.0.0.1:5000/')

def save_att_photos(images):
    output_folder = 'AttendanceCapture'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created directory: {output_folder}")
    else:
        print(f"Directory already exists: {output_folder}")

    count = 0
    for i, image_data in enumerate(images):
        try:
            print(f"Processing image {i + 1}")
            image_data = base64.b64decode(image_data.split(',')[1])
            image_np = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
            if image is None:
                print(f"Failed to decode image {i + 1}")
                continue

            (h, w) = image.shape[:2]
            blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104, 117, 123], False, False)
            net.setInput(blob)
            detections = net.forward()

            for j in range(detections.shape[2]):
                confidence = detections[0, 0, j, 2]
                if confidence > 0.7:
                    box = detections[0, 0, j, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    face = image[startY:endY, startX:endX]
                    cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)

                    count += 1
                    face_filename = os.path.join(output_folder, f'face_{count}.jpg')
                    cv2.imwrite(face_filename, face)
                    print(f'Saved {face_filename}')
                    break
        except Exception as e:
            print(f"Error processing image {i + 1}: {e}")
            continue

    print(f"Total faces saved: {count}")
    return count >= 1
###

def save_photos(student_id, images):
    output_folder = f'Dataset/{student_id}'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created directory: {output_folder}")
    else:
        print(f"Directory already exists: {output_folder}")

    count = 0
    for i, image_data in enumerate(images):
        try:
            print(f"Processing image {i + 1}")
            image_data = base64.b64decode(image_data.split(',')[1])
            image_np = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
            if image is None:
                print(f"Failed to decode image {i + 1}")
                continue

            (h, w) = image.shape[:2]
            blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104, 117, 123], False, False)
            net.setInput(blob)
            detections = net.forward()

            for j in range(detections.shape[2]):
                confidence = detections[0, 0, j, 2]
                if confidence > 0.7:
                    box = detections[0, 0, j, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    face = image[startY:endY, startX:endX]
                    cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)

                    count += 1
                    face_filename = os.path.join(output_folder, f'face_{count}.jpg')
                    cv2.imwrite(face_filename, face)
                    print(f'Saved {face_filename}')
                    break
        except Exception as e:
            print(f"Error processing image {i + 1}: {e}")
            continue

    print(f"Total faces saved: {count}")
    # save_to_csv(student_id, name)
    return count >= 20

def add_student_record(student_id,student_name,student_class):
    csv_file_path = 'Student_record.csv'

    # Check if the student_id already exists in the CSV file to prevent duplicates
    already_registered = False
    if os.path.exists(csv_file_path):
        with open(csv_file_path, mode='r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row['student_id'] == student_id:
                    already_registered = True
                    break

    # If not already registered, append the new student data to the CSV file
    if not already_registered:
        with open(csv_file_path, mode='a', newline='') as csvfile:
            fieldnames = ['student_id', 'student_name', 'student_class']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            # If the file is empty, write the header
            if csvfile.tell() == 0:
                writer.writeheader()

            # Write the student data
            writer.writerow({'student_id': student_id, 'student_name': student_name, 'student_class': student_class})


@app.route('/start_capture', methods=['POST'])
def start_capture():
    data = request.get_json()
    student_id = data.get('student_id')
    images = data.get('images')

    if not student_id or not images:
        return jsonify({'success': False, 'message': 'Invalid data'})

    try:
        if save_photos(student_id, images):
            return jsonify({'success': True, 'message': 'Images saved successfully'})
        else:
            return jsonify({'success': False, 'message': 'Failed to save sufficient images'})

    except Exception as e:
        print(f'Error: {e}')
        return jsonify({'success': False, 'message': 'Failed to save images'})
    


@app.route('/model_training', methods=['POST'])
def model_training():
    success = ModelFineTuning()
    if success:
        return jsonify(success=True, message="Model training completed successfully.")
    else:
        return jsonify(success=False, message="Model training failed.")
    

###
@app.route('/take_photo', methods=['POST'])
def take_photo():
    data = request.get_json()
    # student_id = data.get('student_id')
    images = data.get('images')

    try:
        if save_att_photos(images):
            return jsonify({'success': True, 'message': 'Image saved successfully'})
        else:
            return jsonify({'success': False, 'message': 'Failed to save image'})

    except Exception as e:
        print(f'Error: {e}')
        return jsonify({'success': False, 'message': 'Failed to save image'})


@app.route('/face_prediction', methods=['POST'])
def face_prediction():
    image_path = 'AttendanceCapture/face_1.jpg'
    predicted_label = predict_candidate(image_path)
    print(predicted_label)
    # success = predict_candidate(image_path)
    if predicted_label and str(predicted_label) != '000':
        return jsonify(success=True, predicted_label = predicted_label, message="Attendance Successful")
    elif str(predicted_label) == '000':
        return jsonify(success=False, message="Unknown Face, Please Register Again")
    else:
        return jsonify(success=False, message="Attendance failed")
    
###



@app.route('/check_student_id', methods=['GET'])
def check_student_id():
    student_id = request.args.get('student_id')
    directory_exists = os.path.exists(f'Dataset/{student_id}')
    return jsonify({"directory_exists": directory_exists})



@app.route("/register", methods=["POST", "GET"])
def register():
    if request.method == "POST":
        student_name = request.form["name"]
        student_id = request.form["student_id"]
        student_class = request.form["class_room"]
        if os.path.exists(f'Dataset/{student_id}'):
            return jsonify({"success": False, "message": "Student is already registered."})
        else:
            add_student_record(student_id,student_name,student_class)
            return jsonify({"success": True, "student_id": student_id, "student_name": student_name})
    return render_template("register.html")



@app.route('/capture')
def capture_photos():
    student_id = request.args.get('student_id')
    # name = request.args.get('name')
    print(f"Capturing photos for student_id: {student_id}")
    return render_template("capture_photos.html", student_id=student_id)



@app.route('/attendance')
def attendance():
    return render_template("attendance.html")



@app.route('/')
def home():
    return render_template("index.html")



class TrainingHandler(FileSystemEventHandler):
    def on_modified(self, event):
        print(f'File changed: {event.src_path}')
        if 'custom_gradient.py' in event.src_path:
            print("Restarting training...")
            ModelFineTuning()

if __name__ == "__main__":


    Timer(2, open_browser).start()  # Waits 1 second before opening the web browser
    app.run(debug=True) 

