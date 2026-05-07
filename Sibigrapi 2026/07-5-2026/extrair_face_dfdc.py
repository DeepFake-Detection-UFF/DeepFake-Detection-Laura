import os
import cv2
import json
import random
from tqdm import tqdm

# ---------------------------
# CONFIG
# ---------------------------
VIDEO_DIR = r"C:\Users\laura\OneDrive\Documentos\Dataset\dfdc_train_part_49"
METADATA_PATH = r"C:\Users\laura\OneDrive\Documentos\Dataset\metadata_dfdc_49.json"

OUTPUT_DIR = r"C:\Users\laura\OneDrive\Documentos\Dataset\dfdc_faces"

NUM_FRAMES = 5
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.2
TEST_SPLIT = 0.1

random.seed(42)

# ---------------------------
# LOAD METADATA
# ---------------------------
with open(METADATA_PATH, "r") as f:
    metadata = json.load(f)

videos = list(metadata.keys())

print(f"Total videos: {len(videos)}")

# ---------------------------
# SPLIT DATASET (POR VÍDEO)
# ---------------------------
random.shuffle(videos)

n_total = len(videos)
n_train = int(n_total * TRAIN_SPLIT)
n_val = int(n_total * VAL_SPLIT)

train_videos = videos[:n_train]
val_videos = videos[n_train:n_train + n_val]
test_videos = videos[n_train + n_val:]

print(f"Train: {len(train_videos)}")
print(f"Val: {len(val_videos)}")
print(f"Test: {len(test_videos)}")

# ---------------------------
# FACE DETECTOR (HAAR)
# ---------------------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ---------------------------
# FUNÇÃO: EXTRAI FRAMES
# ---------------------------
def extract_random_frames(video_path, num_frames=5):
    cap = cv2.VideoCapture(video_path)

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if frame_count == 0:
        return []

    frame_idxs = sorted(random.sample(range(frame_count), min(num_frames, frame_count)))

    frames = []
    idx_set = set(frame_idxs)

    current = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if current in idx_set:
            frames.append((current, frame))

        current += 1

    cap.release()
    return frames

# ---------------------------
# FUNÇÃO: DETECTA FACE
# ---------------------------
def detect_face(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    if len(faces) == 0:
        return None

    # pega maior face
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

    face = frame[y:y+h, x:x+w]
    return face

# ---------------------------
# FUNÇÃO: PROCESSAR VÍDEO
# ---------------------------
def process_video(video_name, split):

    video_path = os.path.join(VIDEO_DIR, video_name)

    if not os.path.exists(video_path):
        return

    label = metadata[video_name]["label"]  # REAL ou FAKE

    save_dir = os.path.join(OUTPUT_DIR, split, label)
    os.makedirs(save_dir, exist_ok=True)

    frames = extract_random_frames(video_path, NUM_FRAMES)

    for idx, frame in frames:

        face = detect_face(frame)

        if face is None:
            continue

        filename = f"{video_name.replace('.mp4','')}_frame_{idx}.png"
        save_path = os.path.join(save_dir, filename)

        cv2.imwrite(save_path, face)

# ---------------------------
# EXECUÇÃO
# ---------------------------
print("\nProcessing TRAIN...")
for v in tqdm(train_videos):
    process_video(v, "train")

print("\nProcessing VAL...")
for v in tqdm(val_videos):
    process_video(v, "val")

print("\nProcessing TEST...")
for v in tqdm(test_videos):
    process_video(v, "test")

print("\nDONE!")