import os
import cv2
import gc
import random
from mtcnn import MTCNN

# -------------------------------
# CONFIG
# -------------------------------
BASE_PATH = r"C:\Users\laura\OneDrive\Documentos\Dataset\FaceForensics++_C23"
OUTPUT_BASE = r"C:\Users\laura\OneDrive\Documentos\Dataset\FaceForensics++_faces"

IMG_SIZE = 224
FRAMES_PER_VIDEO = 5

detector = MTCNN()

# -------------------------------
# SPLIT 
# -------------------------------
def split_videos(video_list):
    random.shuffle(video_list)
    n = len(video_list)

    train = video_list[:int(0.7*n)]
    test  = video_list[int(0.7*n):int(0.9*n)]
    val   = video_list[int(0.9*n):]

    return train, test, val

# -------------------------------
# FACE EXTRACTION 
# -------------------------------
def extract_face(image):
    h, w, _ = image.shape

    max_side = max(h, w)
    if max_side > 800:
        scale = 800 / max_side
        image_small = cv2.resize(image, (int(w*scale), int(h*scale)))
    else:
        scale = 1.0
        image_small = image

    results = detector.detect_faces(image_small)
    if not results:
        return None

    best = max(results, key=lambda x: x['box'][2] * x['box'][3])
    x, y, w_box, h_box = best['box']

    x = int(x / scale)
    y = int(y / scale)
    w_box = int(w_box / scale)
    h_box = int(h_box / scale)

    h0, w0, _ = image.shape
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(w0, x + w_box)
    y2 = min(h0, y + h_box)

    if x2 <= x1 or y2 <= y1:
        return None

    face = image[y1:y2, x1:x2]
    if face.size == 0:
        return None

    return cv2.resize(face, (IMG_SIZE, IMG_SIZE))

# -------------------------------
# PROCESS VIDEO
# -------------------------------
def process_video(video_path, save_dir, video_name):
    cap = cv2.VideoCapture(video_path)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return

    frames_to_sample = min(FRAMES_PER_VIDEO, total_frames)
    indices = sorted(random.sample(range(total_frames), frames_to_sample))

    saved = 0

    for i, idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face = extract_face(rgb)

        if face is None:
            continue

        face = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)

        # 🔥 SALVANDO DIRETO (sem pasta do vídeo)
        save_name = f"{video_name}_frame_{i}.png"
        save_path = os.path.join(save_dir, save_name)

        cv2.imwrite(save_path, face)
        saved += 1

    cap.release()
    gc.collect()

# -------------------------------
# MAIN LOOP
# -------------------------------
for class_name in os.listdir(BASE_PATH):

    if class_name.lower() == "csv":
        continue

    class_path = os.path.join(BASE_PATH, class_name)
    if not os.path.isdir(class_path):
        continue

    print(f"\n📁 Classe: {class_name}")

    videos = [v for v in os.listdir(class_path) if v.endswith(".mp4")]

    train, test, val = split_videos(videos)

    splits = {
        "train": train,
        "test": test,
        "val": val
    }

    for split_name, split_videos_list in splits.items():

        print(f"➡️ {split_name}: {len(split_videos_list)} vídeos")

        for video_file in split_videos_list:

            video_path = os.path.join(class_path, video_file)

            save_dir = os.path.join(
                OUTPUT_BASE,
                class_name,
                split_name
            )

            os.makedirs(save_dir, exist_ok=True)

            video_name = os.path.splitext(video_file)[0]

            process_video(video_path, save_dir, video_name)

            gc.collect()