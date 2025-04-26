import os
import cv2
from mtcnn import MTCNN
import random
import tensorflow as tf

path = '[ALTERAR]/face_forensic'
output_dir = '[ALTERAR]/face_forensic//output_frames'
detector = MTCNN()
IMG_SIZE = 224

dir_list = os.listdir(path)

for folder in dir_list:
    folder_path = os.path.join(path, folder)
    print(folder_path)

def extract_face(image):
    result = detector.detect_faces(image)
    if result:
        x, y, w, h = result[0]['box']
        face = image[y:y+h, x:x+w]
        return cv2.resize(face, (IMG_SIZE, IMG_SIZE))
    return cv2.resize(image, (IMG_SIZE, IMG_SIZE))

def video_to_frames(video_path, max_frames=5):
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    random_frames = random.sample(range(total_frames), max_frames)
    random_frames.sort()

    count = 0
    for frame_idx in random_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face = extract_face(frame)
        frames.append(face / 255.0)
        count += 1

    cap.release()
    return frames

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

dir_list = os.listdir(path)

for folder in dir_list:
    folder_path = os.path.join(path, folder)

    if not os.path.isdir(folder_path):
        continue

    print(f'Processando vídeos em: {folder_path}')

    video_files = [f for f in os.listdir(folder_path) if f.endswith('.mp4')]

    for video_file in video_files:
        video_path = os.path.join(folder_path, video_file)
        frames = video_to_frames(video_path)

        for idx, frame in enumerate(frames):
            frame_output_path = os.path.join(output_dir, f'{folder}_{video_file}_frame{idx + 1}.jpg')
            cv2.imwrite(frame_output_path, cv2.cvtColor((frame * 255).astype('uint8'), cv2.COLOR_RGB2BGR))
            print(f'Salvando {frame_output_path}')