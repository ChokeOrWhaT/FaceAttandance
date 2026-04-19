from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import face_recognition
import numpy as np
import base64
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime, date
import logging

# ===============================
# 🔧 CONFIGURATION
# ===============================

app = Flask(__name__)
CORS(app)

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===============================
# 🔥 FIREBASE INITIALIZATION
# ===============================

try:
    if not firebase_admin._apps:
        cred = credentials.Certificate("serviceAccountKey.json")
        firebase_admin.initialize_app(cred)

    db = firestore.client()
    logger.info("✅ Firebase initialized successfully")
except Exception as e:
    logger.error(f"❌ Firebase initialization failed: {e}")
    raise

# ===============================
# 🧠 FACE ENCODING CACHE
# ===============================

known_faces = []
known_names = []
face_encodings_cache = {}


def load_students():
    """Load all student face encodings from Firestore"""
    global known_faces, known_names, face_encodings_cache

    try:
        known_faces = []
        known_names = []
        face_encodings_cache = {}

        students = db.collection("students").stream()

        for doc in students:
            data = doc.to_dict()
            name = data.get("name", "Unknown")
            encoding = data.get("encoding", [])

            if encoding and len(encoding) == 128:  # Valid face encoding size
                known_names.append(name)
                known_faces.append(np.array(encoding))
                face_encodings_cache[name] = {
                    "encoding": np.array(encoding),
                    "id": doc.id,
                }

        logger.info(f"✅ Loaded {len(known_faces)} student encodings")
        return True
    except Exception as e:
        logger.error(f"❌ Error loading students: {e}")
        return False


# Load students on startup
load_students()

# ===============================
# 🔥 HELPER FUNCTIONS
# ===============================


def process_image(image_data):
    """Process base64 image and return RGB frame"""
    try:
        # Handle data URI format
        if isinstance(image_data, str) and image_data.startswith("data:image"):
            image_data = image_data.split(",")[1]

        # Decode base64
        image_bytes = base64.b64decode(image_data)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            raise ValueError("Failed to decode image")

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return rgb_frame, True
    except Exception as e:
        logger.error(f"❌ Image processing error: {e}")
        return None, False


def get_face_encodings(rgb_frame, tolerance=0.6):
    """Extract face encodings from image"""
    try:
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        if len(face_encodings) == 0:
            return None, "no_face"

        return face_encodings[0], "success"
    except Exception as e:
        logger.error(f"❌ Face encoding error: {e}")
        return None, "error"


def is_duplicate_attendance(student_name, today):
    """Check if student already marked attendance today"""
    try:
        existing = (
            db.collection("attendance")
            .where("studentName", "==", student_name)
            .where("date", "==", today)
            .stream()
        )

        return len(list(existing)) > 0
    except Exception as e:
        logger.error(f"❌ Error checking duplicate: {e}")
        return False


# ===============================
# 🔥 API ENDPOINTS
# ===============================


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return (
        jsonify(
            {
                "status": "healthy",
                "students_loaded": len(known_faces),
                "timestamp": datetime.now().isoformat(),
            }
        ),
        200,
    )


@app.route("/register-face", methods=["POST"])
def register_face():
    try:
        data = request.get_json()

        if not data:
            return jsonify({"status": "error", "message": "Invalid JSON data"}), 400

        name = data.get("name", "").strip().lower()
        image_data = data.get("image", "").strip()

        if not name:
            return (
                jsonify({"status": "error", "message": "Valid name is required"}),
                400,
            )

        if not image_data:
            return (
                jsonify({"status": "error", "message": "Valid image is required"}),
                400,
            )

        rgb_frame, success = process_image(image_data)
        if not success:
            return jsonify({"status": "error", "message": "Invalid image format"}), 400

        encoding, status = get_face_encodings(rgb_frame)
        if status != "success":
            return (
                jsonify({"status": "error", "message": "No clear face detected"}),
                400,
            )

        # Save to Firestore
        db.collection("students").document(name).set(
            {
                "name": name,
                "encoding": encoding.tolist(),
                "registered_at": datetime.now(),
            }
        )

        # Reload memory cache
        load_students()

        return (
            jsonify(
                {
                    "status": "success",
                    "message": f"Face registered successfully for {name}",
                }
            ),
            200,
        )

    except Exception as e:
        logger.error(f"Registration error: {e}")
        return (
            jsonify({"status": "error", "message": "Server error during registration"}),
            500,
        )


@app.route("/scan-face", methods=["POST"])
def scan_face():
    try:
        data = request.get_json()

        if not data:
            return jsonify({"status": "error", "message": "Invalid JSON data"}), 400

        image_data = data.get("image")
        if not image_data or image_data.strip() == "":
            return jsonify({"status": "error", "message": "Image is required"}), 400

        # Process image
        rgb_frame, success = process_image(image_data)
        if not success:
            return jsonify({"status": "error", "message": "Invalid image format"}), 400

        # Extract encoding
        encoding, status = get_face_encodings(rgb_frame)
        if status != "success":
            return (
                jsonify({"status": "error", "message": "No clear face detected"}),
                400,
            )

        # Make sure we have registered faces
        if len(known_faces) == 0:
            return (
                jsonify({"status": "error", "message": "No registered faces found"}),
                400,
            )

        # Compare
        face_distances = face_recognition.face_distance(known_faces, encoding)
        best_match_index = np.argmin(face_distances)
        best_match_distance = face_distances[best_match_index]

        print("Best match distance:", best_match_distance)

        RECOGNITION_THRESHOLD = 0.6

        if best_match_distance > RECOGNITION_THRESHOLD:
            return jsonify({"status": "error", "message": "Face not recognized"}), 400

        matched_name = known_names[best_match_index]

        # Save attendance
        db.collection("attendance").add(
            {
                "studentName": matched_name,
                "date": datetime.now().strftime("%Y-%m-%d"),
                "time": datetime.now().strftime("%H:%M:%S"),
                "confidence": float(1 - best_match_distance),
                "marked_at": datetime.now(),
            }
        )

        logger.info(f"✅ Attendance marked for {matched_name}")

        return (
            jsonify(
                {
                    "status": "success",
                    "name": matched_name,
                    "confidence": float(1 - best_match_distance),
                }
            ),
            200,
        )

    except Exception as e:
        logger.error(f"Scan error: {e}")
        return (
            jsonify({"status": "error", "message": "Server error during scanning"}),
            500,
        )


@app.route("/update-face", methods=["POST"])
def update_face():
    """Update a student's face encoding"""
    try:
        data = request.json
        name = data.get("name", "").strip().lower()
        image_data = data.get("image", "").strip()

        if not name or not image_data:
            return (
                jsonify({"status": "error", "message": "Name and image are required"}),
                400,
            )

        # Check if student exists
        doc = db.collection("students").document(name).get()
        if not doc.exists():
            return jsonify({"status": "error", "message": "Student not found"}), 404

        # Process image
        rgb_frame, success = process_image(image_data)
        if not success:
            return jsonify({"status": "error", "message": "Invalid image format"}), 400

        # Extract face encoding
        encoding, status = get_face_encodings(rgb_frame)

        if status != "success":
            return (
                jsonify(
                    {
                        "status": "no_face_detected",
                        "message": "No clear face detected in image",
                    }
                ),
                400,
            )

        # Update in Firestore
        db.collection("students").document(name).update(
            {"encoding": encoding.tolist(), "updated_at": datetime.now()}
        )

        # Reload cache
        load_students()

        logger.info(f"✅ Face updated for {name}")

        return (
            jsonify(
                {
                    "status": "success",
                    "message": f"Face updated successfully for {name}",
                }
            ),
            200,
        )

    except Exception as e:
        logger.error(f"❌ Update error: {e}")
        return (
            jsonify({"status": "error", "message": "Server error during update"}),
            500,
        )


@app.route("/reload-cache", methods=["POST"])
def reload_cache():
    """Reload student face encodings cache"""
    try:
        success = load_students()

        if success:
            return (
                jsonify(
                    {
                        "status": "success",
                        "students_loaded": len(known_faces),
                        "message": "Cache reloaded successfully",
                    }
                ),
                200,
            )
        else:
            return (
                jsonify({"status": "error", "message": "Failed to reload cache"}),
                500,
            )
    except Exception as e:
        logger.error(f"❌ Reload error: {e}")
        return jsonify({"status": "error", "message": "Server error"}), 500


@app.route("/stats", methods=["GET"])
def get_stats():
    """Get attendance statistics"""
    try:
        # Get total students
        students = db.collection("students").stream()
        total_students = len(list(students))

        # Get today's attendance
        today = datetime.now().strftime("%Y-%m-%d")
        attendance_docs = (
            db.collection("attendance").where("date", "==", today).stream()
        )
        present_today = len(list(attendance_docs))

        # Get total attendance records
        all_attendance = db.collection("attendance").stream()
        total_records = len(list(all_attendance))

        return (
            jsonify(
                {
                    "status": "success",
                    "stats": {
                        "total_students": total_students,
                        "present_today": present_today,
                        "total_attendance_records": total_records,
                        "students_loaded": len(known_faces),
                    },
                }
            ),
            200,
        )
    except Exception as e:
        logger.error(f"❌ Stats error: {e}")
        return jsonify({"status": "error", "message": "Server error"}), 500


# ===============================
# ❌ ERROR HANDLERS
# ===============================


@app.errorhandler(404)
def not_found(error):
    return jsonify({"status": "error", "message": "Endpoint not found"}), 404


@app.errorhandler(500)
def server_error(error):
    return jsonify({"status": "error", "message": "Internal server error"}), 500


# ===============================
# 🚀 SERVER STARTUP
# ===============================

if __name__ == "__main__":
    logger.info("🚀 Starting FaceScan AI Backend Server")
    logger.info("📍 Running on http://0.0.0.0:5000")
    logger.info("✅ CORS enabled for cross-origin requests")

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,  # Set to True for development
        threaded=True,
    )
