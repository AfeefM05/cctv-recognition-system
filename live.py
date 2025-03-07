import cv2 as cv
import numpy as np
import insightface
import faiss
import pickle
import time
import threading
from queue import Queue
from db import FaceDBLoggerSQL

# Initialize InsightFace model
model = insightface.app.FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])
model.prepare(ctx_id=0, det_size=(640, 640))

#Initialise db
database=FaceDBLoggerSQL(host="127.0.0.1",port=3306)

# Load embeddings and labels
with open('labels.pkl', 'rb') as f:
    data = pickle.load(f)

Y = data['labels']

# Setup FAISS IndexFlatIP
index = faiss.read_index('face_index.faiss')

# Camera URLs
camera_urls = [
    0
]

# Desired resolution for processing/display
desired_width = 640
desired_height = 480

# Data structures for each camera
camera_data = {}
for i, url in enumerate(camera_urls):
    camera_data[i] = {
        'frame_queue': Queue(maxsize=2),
        'result_queue': Queue(maxsize=2),
        'last_result': None,
        'last_result_time': 0,
        'display_fps': 0,
        'processing_fps': 0,
        'frame_times': [],
        'processing_times': [],
        'video': None,
        'running': True
    }

# Face processing thread
def process_frames(camera_id):
    data = camera_data[camera_id]
    result_queue = data['result_queue']  # Get the camera-specific result queue
    while data['running']:
        try:
            frame = data['frame_queue'].get(timeout=0.3)

            # Resize frame to desired resolution for processing
            resized_frame = cv.resize(frame, (desired_width, desired_height))

            result = {
                'faces': [],
                'frame_time': time.time()
            }

            faces = model.get(resized_frame)  # Use resized_frame for face detection

            if faces:
                embeddings = []
                face_data = []

                for face in faces:
                    bbox = face.bbox.astype(int)
                    embedding = face.embedding.astype(np.float32).reshape(1, -1)
                    embedding /= np.linalg.norm(embedding)  # Ensure normalized (cosine/IP fix)

                    embeddings.append(embedding[0])
                    face_data.append((face, bbox))

                if embeddings:
                    embeddings = np.array(embeddings, dtype=np.float32)
                    distances, indices = index.search(embeddings, k=1)

                    for (face, bbox), distance, idx in zip(face_data, distances, indices):
                        matched_name = Y[idx[0]] if distance[0] > 0.5 else "Unknown"
                        color = (0, 255, 0) if matched_name != "Unknown" else (0, 0, 255)

                        result['faces'].append({
                            'bbox': bbox,
                            'distance': distance[0],
                            'name': matched_name,
                            'color': color
                        })
                        if distance[0]>0.5 and database.should_log_detection(matched_name):
                            database.log_detection(matched_name,float(distance[0]),camera_id)

            result_queue.put(result)
            data['frame_queue'].task_done()

        except Exception as e:
            print(f"Camera {camera_id} Processing error: {e}")

# Initialize and start threads
threads = []
for i in camera_data:
    thread = threading.Thread(target=process_frames, args=(i,), daemon=True)
    threads.append(thread)
    camera_data[i]['running'] = True  # Ensure the running flag is set before starting
    thread.start()

# Video capture setup
for i, url in enumerate(camera_urls):
    data = camera_data[i]
    data['video'] = cv.VideoCapture(url)

# Helper function to draw labels on the frame
def draw_label(img, text, pos, color, scale=0.7, thickness=2):
    """Improved text rendering for readability."""
    cv.putText(img, text, pos, cv.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness * 3)
    cv.putText(img, text, pos, cv.FONT_HERSHEY_SIMPLEX, scale, color, thickness)

while True:
    frames = []
    for i in camera_data:
        data = camera_data[i]
        ret, frame = data['video'].read()
        if not ret:
            data['running'] = False
            continue

        # Resize frame to desired resolution for display
        resized_frame = cv.resize(frame, (desired_width, desired_height))

        current_time = time.time()

        # FPS - Display
        data['frame_times'].append(current_time)
        data['frame_times'] = [t for t in data['frame_times'] if t > current_time - 1]
        data['display_fps'] = len(data['frame_times'])

        if not data['frame_queue'].full():
            data['frame_queue'].put(frame.copy())  # Put the original frame, not resized

        # Retrieve latest result if available
        while not data['result_queue'].empty():
            data['last_result'] = data['result_queue'].get()
            data['last_result_time'] = current_time


        # Draw latest face results (if recent)
        if data['last_result'] and current_time - data['last_result_time'] < 0.5:
            data['processing_times'].append(data['last_result']['frame_time'])
            data['processing_times'] = [t for t in data['processing_times'] if t > current_time - 1]
            data['processing_fps'] = len(data['processing_times'])

            for face in data['last_result']['faces']:
                bbox = face['bbox']
                name = face['name']
                distance = face['distance']
                color = face['color']

                cv.rectangle(resized_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

                label = f"{name} ({distance:.2f})"
                draw_label(resized_frame, label, (bbox[0], bbox[1] - 10), color)


        # Display FPS overlay
        fps_text = f"Cam {i} Display FPS: {data['display_fps']} | Processing FPS: {data['processing_fps']}"
        draw_label(resized_frame, fps_text, (10, 30), (0, 255, 0))

        frames.append(resized_frame)

    # Concatenate frames horizontally to display side by side
    if frames:
        if len(frames)==4:
            c1=np.hstack((frames[0],frames[1]))
            c2=np.hstack((frames[2],frames[3]))
            combined_frame=np.vstack((c1,c2))
        else:
            combined_frame=np.hstack(frames)

        # Show the combined frame with all cameras' outputs side by side
        cv.imshow('Combined Camera Feed', combined_frame)

    if cv.waitKey(1) & 0xFF in (ord('q'), 27):
        break

# Clean up
for i in camera_data:
    camera_data[i]['running'] = False

for thread in threads:
    thread.join(timeout=1.0)

for i in camera_data:
    data = camera_data[i]
    if data['video']:
        data['video'].release()

cv.destroyAllWindows()
