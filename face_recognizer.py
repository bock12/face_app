import cv2
import os
import face_recognition
import datetime
import pickle
import time
from face_db import connect_db

class FaceRecognizerDL:
    def __init__(self, known_encodings, known_names, unknown_dir, cooldown_second=60):
        self.known_encodings = known_encodings
        self.known_names = known_names
        self.unknown_dir = unknown_dir
        self.cooldown_second = cooldown_second
        self.PAD_FRAC = 0.2
        os.makedirs(self.unknown_dir, exist_ok=True)

        self.load_known_faces()
        self.cooldowns = {}  # Map: encoding_id -> last alert time

    def load_known_faces(self):
        self.known_encodings.clear()
        self.known_names.clear()

        conn = connect_db()
        c = conn.cursor()
        c.execute("SELECT name, encoding FROM known_faces")
        for name, enc_blob in c.fetchall():
            self.known_names.append(name)
            self.known_encodings.append(pickle.loads(enc_blob))
        conn.close()

    def detect_and_recognize(self, frame, send_alert):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        facelocs = face_recognition.face_locations(rgb, model="hog")
        encodings = face_recognition.face_encodings(rgb, facelocs)

        for (top, right, bottom, left), encoding in zip(facelocs, encodings):
            matches = face_recognition.compare_faces(self.known_encodings, encoding)
            name = "Unknown"

            if True in matches:
                matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                counts = {}
                for i in matchedIdxs:
                    name = self.known_names[i]
                    counts[name] = counts.get(name, 0) + 1
                name = max(counts, key=counts.get)

            fw = right - left
            fh = bottom - top
            pad_x = int(fw * self.PAD_FRAC)
            pad_y = int(fh * self.PAD_FRAC)

            h, w = frame.shape[:2]
            left_p = max(left - pad_x, 0)
            top_p = max(top - pad_y, 0)
            right_p = min(right + pad_x, w)
            bottom_p = min(bottom + pad_y, h)

            if name == "Unknown":
                self.handle_unknown_face(frame, encoding, top_p, right_p, bottom_p, left_p, send_alert)

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        return frame

    def handle_unknown_face(self, frame, encoding, top_p, right_p, bottom_p, left_p, send_alert):
        conn = connect_db()
        c = conn.cursor()

        c.execute("SELECT id, encoding FROM unknown_faces")
        unknowns = c.fetchall()
        conn.close()

        min_dist = float("inf")
        min_id = None

        for uid, enc_blob in unknowns:
            stored_encoding = pickle.loads(enc_blob)
            dist = face_recognition.face_distance([stored_encoding], encoding)[0]
            if dist < min_dist:
                min_dist = dist
                min_id = uid

        TOLERANCE = 0.5
        now = time.time()

        if min_dist > TOLERANCE or not unknowns:
            # New unknown face
            face_crop = frame[top_p:bottom_p, left_p:right_p]
            try:
                passport = cv2.resize(face_crop, (300, 300))
            except:
                passport = face_crop

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"unknown_{timestamp}.jpg"
            image_path = os.path.join(self.unknown_dir, filename)
            cv2.imwrite(image_path, passport)

            conn = connect_db()
            c = conn.cursor()
            c.execute("INSERT INTO unknown_faces (image_path, encoding, date_detected) VALUES (?, ?, ?)",
                      (image_path, pickle.dumps(encoding), datetime.datetime.now().isoformat()))
            conn.commit()
            conn.close()

            send_alert("Unknown", image_path)

        else:
            # Existing face, check cooldown
            if min_id not in self.cooldowns or (now - self.cooldowns[min_id] > self.cooldown_second):
                self.cooldowns[min_id] = now

                conn = connect_db()
                c = conn.cursor()
                c.execute("SELECT image_path FROM unknown_faces WHERE id = ?", (min_id,))
                row = c.fetchone()
                conn.close()

                if row:
                    send_alert("Unknown", row[0])
            else:
                print("[INFO] Skipped duplicate within cooldown")

