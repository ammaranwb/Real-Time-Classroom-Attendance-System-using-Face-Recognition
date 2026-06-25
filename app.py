import cv2
import csv
import io
import numpy as np
from flask import (
    Flask, render_template, request, redirect,
    url_for, Response, jsonify, flash, send_file
)
from database import init_db, add_user, get_all_users, get_user_by_name, \
                     mark_attendance, get_attendance, delete_user
from face_utils import save_face_image, load_known_faces, recognize_faces, train_model

app = Flask(__name__)
app.secret_key = "attendance_secret_123"

# ── Init DB and load known faces on startup ───────────────────────────────────
init_db()
known_faces = load_known_faces()   # returns (recognizer, label_map)

# ── Camera state ──────────────────────────────────────────────────────────────
camera      = None
recognizing = False


def get_camera():
    global camera
    if camera is None or not camera.isOpened():
        camera = cv2.VideoCapture(0)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return camera


def release_camera():
    global camera
    if camera and camera.isOpened():
        camera.release()
    camera = None


# ══════════════════════════════════════════════════════════════════════════════
#  ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    users      = get_all_users()
    today_logs = get_attendance()[:10]
    return render_template("index.html", users=users, logs=today_logs)


# ── ENROLL ────────────────────────────────────────────────────────────────────
@app.route("/enroll", methods=["GET", "POST"])
def enroll():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        if not name:
            flash("Please enter a name.", "danger")
            return redirect(url_for("enroll"))

        existing = get_user_by_name(name)
        if existing:
            flash(f"'{name}' is already enrolled.", "warning")
            return redirect(url_for("enroll"))

        cap = get_camera()
        ret, frame = cap.read()
        release_camera()

        if not ret:
            flash("Could not access camera. Check connection.", "danger")
            return redirect(url_for("enroll"))

        success, result = save_face_image(name, frame)
        if not success:
            flash(f"Enrollment failed: {result}", "danger")
            return redirect(url_for("enroll"))

        add_user(name, result)

        # Retrain model with new face
        global known_faces
        known_faces = train_model()

        flash(f"'{name}' enrolled successfully!", "success")
        return redirect(url_for("index"))

    return render_template("enroll.html")


# ── ENROLL CAMERA FEED ────────────────────────────────────────────────────────
def enroll_stream():
    cap = get_camera()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2
        bw, bh = 200, 240
        cv2.rectangle(frame,
                      (cx - bw//2, cy - bh//2),
                      (cx + bw//2, cy + bh//2),
                      (0, 255, 0), 2)
        cv2.putText(frame, "Position face in box", (cx - 110, cy - bh//2 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        _, jpeg = cv2.imencode(".jpg", frame)
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" +
               jpeg.tobytes() + b"\r\n")


@app.route("/enroll_feed")
def enroll_feed():
    return Response(enroll_stream(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


# ── ATTENDANCE STREAM ─────────────────────────────────────────────────────────
def attendance_stream():
    global recognizing, known_faces
    cap = get_camera()
    frame_count = 0

    while recognizing:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        if frame_count % 5 == 0:
            detections = recognize_faces(frame, known_faces)

            for det in detections:
                x, y, w, h = det["x"], det["y"], det["w"], det["h"]
                name  = det["name"]
                conf  = det["confidence"]
                color = (0, 200, 0) if name != "Unknown" else (0, 0, 220)

                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                label = f"{name} ({conf}%)"
                cv2.rectangle(frame, (x, y - 28), (x + w, y), color, -1)
                cv2.putText(frame, label, (x + 4, y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

                if name != "Unknown":
                    user = get_user_by_name(name)
                    if user:
                        mark_attendance(user["id"], name, str(conf))

        _, jpeg = cv2.imencode(".jpg", frame)
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" +
               jpeg.tobytes() + b"\r\n")

    release_camera()


@app.route("/attendance")
def attendance():
    return render_template("attendance.html")


@app.route("/stop_recognition", methods=["POST"])
def stop_recognition():
    global recognizing
    recognizing = False
    return jsonify({"status": "stopped"})


@app.route("/attendance_feed")
def attendance_feed():
    global recognizing
    recognizing = True
    return Response(attendance_stream(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


# ── RECORDS ───────────────────────────────────────────────────────────────────
@app.route("/records")
def records():
    filter_date = request.args.get("date", "")
    logs = get_attendance(filter_date if filter_date else None)
    return render_template("records.html", logs=logs, filter_date=filter_date)


@app.route("/export_csv")
def export_csv():
    filter_date = request.args.get("date", "")
    logs = get_attendance(filter_date if filter_date else None)
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["ID", "Name", "Date", "Time", "Confidence"])
    for log in logs:
        writer.writerow([log["id"], log["user_name"],
                         log["date"], log["time"], log["confidence"]])
    output.seek(0)
    return send_file(
        io.BytesIO(output.getvalue().encode()),
        mimetype="text/csv",
        as_attachment=True,
        download_name="attendance.csv"
    )


# ── DELETE USER ───────────────────────────────────────────────────────────────
@app.route("/delete_user/<int:user_id>", methods=["POST"])
def remove_user(user_id):
    delete_user(user_id)
    global known_faces
    known_faces = train_model()
    flash("User deleted.", "info")
    return redirect(url_for("index"))


# ── TODAY COUNT ───────────────────────────────────────────────────────────────
@app.route("/today_count")
def today_count():
    from datetime import date
    today = date.today().strftime("%Y-%m-%d")
    logs  = get_attendance(filter_date=today)
    return jsonify({"count": len(logs)})


if __name__ == "__main__":
    app.run(debug=True)