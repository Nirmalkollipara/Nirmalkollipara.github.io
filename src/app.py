from flask import Flask, request, render_template, jsonify
from ultralytics import YOLO
import os
import time
import cv2

app = Flask(__name__)
model = YOLO('yolov8x.pt')  # Using a larger model for better accuracy

UPLOAD_FOLDER = "D:\\Project\\uploads"
RESULT_FOLDER = 'D:\\Project\\src\\static\\output'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No selected file'})
    
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    if file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        path = process_image(filepath)
        input_type = "image"
    elif file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        path = process_video(filepath)
        input_type = "video"
    else:
        return jsonify({'success': False, 'error': 'Unsupported file type'})
    print(path)
    filepath = path.replace("D:\\Project\\src\\static\\", "")
    filepath = filepath.replace("\\", "/")
    print(filepath)

    return render_template("result.html", filepath=filepath, input_type=input_type)

def process_image(image_path):
    results = model.predict(image_path, save=True, conf=0.3, iou=0.5)
    base, ext = os.path.splitext(image_path)
    result_path = os.path.join(RESULT_FOLDER, f"opimage{ext}")
    
    for result in results:
        result.save(filename=result_path)
        print(result_path,"File Saved!!!")
    
    return result_path

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    base, ext = os.path.splitext(video_path)
    result_path = os.path.join(RESULT_FOLDER, f"opvideo{ext}")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(result_path, fourcc, fps, (width, height))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model.predict(frame, conf=0.3, iou=0.5)
        
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = int(box.cls.item())
                conf = float(box.conf.item())
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'Class {label}: {conf:.2f}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        out.write(frame)

    cap.release()
    out.release()

    return result_path

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
