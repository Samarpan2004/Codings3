import face_recognition
import cv2
import os
import pickle

KNOWN_DIR = "known_faces"  # folder with subfolders per person or images named person_name.jpg
MODEL_FILE = "face_encodings.pkl"

def build_known_encodings(known_dir=KNOWN_DIR):
    encodings = {}
    for fname in os.listdir(known_dir):
        if fname.startswith('.'): continue
        path = os.path.join(known_dir, fname)
        if os.path.isdir(path):
            # folder per person
            person = os.path.basename(path)
            encodings.setdefault(person, [])
            for f in os.listdir(path):
                fp = os.path.join(path, f)
                img = face_recognition.load_image_file(fp)
                face_locs = face_recognition.face_locations(img)
                if not face_locs: continue
                enc = face_recognition.face_encodings(img, known_face_locations=face_locs)[0]
                encodings[person].append(enc)
        else:
            # file naming convention: name.jpg
            person = os.path.splitext(fname)[0]
            img = face_recognition.load_image_file(path)
            locs = face_recognition.face_locations(img)
            if not locs: continue
            enc = face_recognition.face_encodings(img, known_face_locations=locs)[0]
            encodings.setdefault(person, []).append(enc)
    # flatten to list
    known_encs = []
    known_names = []
    for name, encs in encodings.items():
        for e in encs:
            known_encs.append(e)
            known_names.append(name)
    with open(MODEL_FILE, "wb") as f:
        pickle.dump({"encodings": known_encs, "names": known_names}, f)
    print(f"Saved {len(known_encs)} face encodings to {MODEL_FILE}")
    return known_encs, known_names

def load_known_encodings():
    if not os.path.exists(MODEL_FILE):
        return build_known_encodings()
    with open(MODEL_FILE, "rb") as f:
        data = pickle.load(f)
    return data["encodings"], data["names"]

def run_webcam_recognition():
    known_encodings, known_names = load_known_encodings()
    video_capture = cv2.VideoCapture(0)
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        small = cv2.resize(frame, (0,0), fx=0.25, fy=0.25)
        rgb_small = small[:, :, ::-1]
        face_locs = face_recognition.face_locations(rgb_small)
        face_encs = face_recognition.face_encodings(rgb_small, face_locs)
        for (top, right, bottom, left), enc in zip(face_locs, face_encs):
            matches = face_recognition.compare_faces(known_encodings, enc, tolerance=0.5)
            name = "Unknown"
            if True in matches:
                first_match = matches.index(True)
                name = known_names[first_match]
            # scale back up
            top *= 4; right *= 4; bottom *= 4; left *= 4
            cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)
        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Build encodings if needed, then run webcam
    if not os.path.exists(MODEL_FILE):
        build_known_encodings()
    run_webcam_recognition()
