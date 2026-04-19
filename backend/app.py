from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import face_recognition
import numpy as np
import base64
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime

app = Flask(__name__)
CORS(app)

# ======================================
# 🔥 FIREBASE INITIALIZATION
# ======================================
if not firebase_admin._apps:
    cred = credentials.Certificate("serviceAccountKey.json")
    firebase_admin.initialize_app(cred)

db = firestore.client()

# ======================================
# 🔥 LOAD STORED STUDENTS
# ======================================
known_faces = []
known_names = []

def load_students():
    global known_faces, known_names
    known_faces = []
    known_names = []

    students = db.collection("students").stream()

    for doc in students:
        data = doc.to_dict()

        if "name" in data and "encoding" in data:
            known_names.append(data["name"])
            known_faces.append(np.array(data["encoding"]))

load_students()

# ======================================
# 🔥 REGISTER FACE
# ======================================
@app.route("/register-face", methods=["POST"])
def register_face():
    data = request.json
    name = data.get("name")
    image_data = data.get("image")

    if not name or not image_data:
        return jsonify({"status": "invalid data"}), 400

    try:
        image_bytes = base64.b64decode(image_data.split(",")[1])
        np_arr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        if len(face_encodings) == 0:
            return jsonify({"status": "no face detected"})

        encoding = face_encodings[0]

        db.collection("students").document(name).set({
            "name": name,
            "encoding": encoding.tolist()
        })

        load_students()

        return jsonify({"status": "registered successfully"})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# ======================================
# 🔥 SCAN FACE
# ======================================
@app.route("/scan-face", methods=["POST"])
def scan_face():
    data = request.json
    image_data = data.get("image")

    if not image_data:
        return jsonify({"status": "no image provided"}), 400

    try:
        image_bytes = base64.b64decode(image_data.split(",")[1])
        np_arr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        if len(face_encodings) == 0:
            return jsonify({"status": "no face detected"})

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(
                known_faces,
                face_encoding,
                tolerance=0.5
            )

            if True in matches:
                match_index = matches.index(True)
                name = known_names[match_index]

                today = datetime.now().strftime("%Y-%m-%d")

                existing = db.collection("attendance") \
                    .where("studentName", "==", name) \
                    .where("date", "==", today) \
                    .stream()

                if list(existing):
                    return jsonify({
                        "status": "already marked",
                        "name": name
                    })

                db.collection("attendance").add({
                    "studentName": name,
                    "date": today,
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "status": "Present"
                })

                return jsonify({
                    "status": "success",
                    "name": name
                })

        return jsonify({"status": "not recognized"})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# ======================================
# 🚀 RUN SERVER
# ======================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)