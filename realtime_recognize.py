import time
import numpy as np
import cv2
from insightface.app import FaceAnalysis

DB_PATH = "db/criminals.npz"

# Cosine similarity threshold:
# 0.30–0.40 (strict), 0.40–0.55 (balanced), >0.55 (looser)
SIM_THRESHOLD = 0.45

def l2_normalize(v, eps=1e-10):
    return v / max(np.linalg.norm(v), eps)

def cosine_sim(a, b):
    return float(np.dot(a, b))  # embeddings are L2-normalized

def load_db(path):
    data = np.load(path, allow_pickle=True)
    names = data["names"]
    embs  = data["embeddings"].astype(np.float32)
    # Ensure normalized (should already be):
    embs = np.stack([l2_normalize(e) for e in embs], axis=0)
    return names, embs

def init_face_app():
    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=-1, det_size=(640, 640))  # CPU
    return app

def annotate(frame, bbox, label, color=(0,255,0)):
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
    cv2.putText(frame, label, (x1, max(0,y1-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

def main():
    names, db_embs = load_db(DB_PATH)
    app = init_face_app()

    cap = cv2.VideoCapture(0)  # change index if needed
    if not cap.isOpened():
        print("[ERROR] Cannot open camera.")
        return

    fps_t0 = time.time()
    frames = 0

    print("[INFO] Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = app.get(frame)
        for f in faces:
            emb = f.normed_embedding
            if emb is None:
                emb = l2_normalize(f.embedding.astype(np.float32))

            # Compute cosine similarity to each enrolled identity
            sims = db_embs @ emb  # (N,256) x (256,) => (N,)
            idx = int(np.argmax(sims))
            best_sim = float(sims[idx])

            if best_sim >= SIM_THRESHOLD:
                label = f"{names[idx]} ({best_sim:.2f})"
                color = (0,0,255)  # red for match (criminal)
            else:
                label = f"Unknown ({best_sim:.2f})"
                color = (255,255,0)

            annotate(frame, f.bbox, label, color=color)

        frames += 1
        if frames % 20 == 0:
            dt = time.time() - fps_t0
            fps = frames / dt if dt > 0 else 0
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)

        cv2.imshow("Criminal Face Detection - InsightFace", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
