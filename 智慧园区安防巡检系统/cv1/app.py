import cv2 as cv
from ultralytics import YOLO
from paddleocr import PaddleOCR
import os
import numpy as np
import threading
from flask import Flask, render_template, jsonify, Response, request, session
from flask_cors import CORS
import sqlite3 as sql
import time
from datetime import datetime

fire_model = YOLO(r"C:\Users\39608\Desktop\demo1\models\fire_best.pt")
car_model = YOLO(r"C:\Users\39608\Desktop\demo1\models\vehicle.pt")
plate_model = YOLO(r"C:\Users\39608\Desktop\demo1\models\plate.pt")
face_model = YOLO(r"C:\Users\39608\Desktop\demo1\models\face_best5.pt")

face_conf = 0.5
fire_conf = 0.5
water_conf = 0.8

# ocr = PaddleOCR(
#     use_angle_cls=True,
#     lang='ch',
#     det_model_dir=r'inference/ch_ppocr_mobile_v2.0_det_infer',
#     rec_model_dir=r'inference/ch_ppocr_mobile_v2.0_rec_infer'
# )


num = 0
num_lock = threading.Lock()
db_lock = threading.Lock() 

app = Flask(__name__)


SAVE_DIR = "snapshots"
os.makedirs(SAVE_DIR, exist_ok=True)


def get_db_connection():
    """获取数据库连接"""
    conn = sql.connect('show.db', check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute("PRAGMA journal_mode = WAL")
    return conn, cursor


conn, cursor = get_db_connection()
cursor.execute('''
    CREATE TABLE IF NOT EXISTS alerts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        alert_type TEXT,
        confidence REAL,
        description TEXT
    )
''')

cursor.execute('''
    CREATE TABLE IF NOT EXISTS alert_images (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        alert_id INTEGER,
        timestamp TEXT NOT NULL,
        image_data BLOB,
        file_path TEXT,
        FOREIGN KEY (alert_id) REFERENCES alerts(id) ON DELETE CASCADE
    )
''')
conn.commit()
conn.close()

MIN_SAVE_INTERVAL = 3.0
last_save_time = {}

def save_snapshot(frame, alert_type, confidence):
    """保存检测截图（带防抖机制）"""
    global last_save_time

    # 防抖检查
    current_time = time.time()
    if alert_type in last_save_time:
        if current_time - last_save_time[alert_type] < MIN_SAVE_INTERVAL:
            return None

    # 生成文件名
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    filename = f"{alert_type}_{confidence:.2f}_{timestamp}.jpg"
    full_path = os.path.join(SAVE_DIR, filename)

    # 保存图片到文件
    cv.imwrite(full_path, frame, [cv.IMWRITE_JPEG_QUALITY, 90])

    # 保存到数据库（加锁保证线程安全）
    with db_lock:
        try:
            conn, cursor = get_db_connection()

            # 编码为BLOB
            success, encoded = cv.imencode('.jpg', frame, [cv.IMWRITE_JPEG_QUALITY, 85])
            if success:
                image_blob = encoded.tobytes()

                # 插入主表
                cursor.execute('''
                    INSERT INTO alerts (timestamp, alert_type, confidence, description)
                    VALUES (?, ?, ?, ?)
                ''', (datetime.now().isoformat(), alert_type, confidence, f"检测到{alert_type}"))

                alert_id = cursor.lastrowid

                # 插入图片表
                cursor.execute('''
                    INSERT INTO alert_images (alert_id, timestamp, image_data, file_path)
                    VALUES (?, ?, ?, ?)
                ''', (alert_id, datetime.now().isoformat(), image_blob, full_path))

                conn.commit()
                conn.close()

                print(f"🚨 截图保存: {alert_type} (置信度: {confidence:.2f})=={filename}")

        except Exception as e:
            print(f"保存到数据库失败: {e}")
            try:
                conn.rollback()
                conn.close()
            except:
                pass

    last_save_time[alert_type] = current_time
    return full_path


# 人脸检测
def face_recognize(frame):
    global num
    result = face_model(frame, verbose=False)

    for s in result:
        boxes = s.boxes
        for box in boxes:
            conf = box.conf[0]
            cls_id = int(box.cls[0])
            label = face_model.names[cls_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if conf < face_conf:
                cv.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv.putText(frame, 'Stranger', (x1, y1), cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

                with num_lock:
                    num += 1

                save_snapshot(frame, "Stranger_Face", float(conf))

            else:
                cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv.putText(frame, label, (x1, y1), cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    return frame

# 火灾检测
def fire_recognize(frame):
    global num
    result = fire_model(frame, verbose=False)

    boxes = result[0].boxes
    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            cv_conf = box.conf[0]
            if cv_conf > fire_conf:
                cv.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                cv.putText(frame, "Fire", (int(x1), int(y1)-10), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                with num_lock:
                    num += 1

              
                save_snapshot(frame, "Fire", float(cv_conf))

    return frame

# 积水检测
# def water_recognize(frame):
#     global num
#     results = water_model(frame, verbose=False)
#     boxes = results[0].boxes
#     if boxes is not None and len(boxes) > 0:
#         for box in boxes:
#             x1, y1, x2, y2 = box.xyxy[0]
#             cv_conf = box.conf[0]
#             if cv_conf > water_conf:
#                 cv.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
#                 cv.putText(frame, "Ponding", (int(x1), int(y1)-10), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

#                 with num_lock:
#                     num += 1

#                 save_snapshot(frame, "Ponding", float(cv_conf))

#     return frame

# 车辆识别
# WHITELIST = {"粤A3306D", "京A00001"}

# def car_recognize(frame):
#     global num
#     stranger_alert = False
#     results = car_model(frame, verbose=False)

#     for c in results:
#         if c.boxes is not None and len(c.boxes) > 0:
#             for box in c.boxes:
#                 x1, y1, x2, y2 = map(int, box.xyxy[0])
#                 roi = frame[y1:y2, x1:x2]

#                 plate_results = plate_model(roi, verbose=False)
#                 for p in plate_results:
#                     if p.boxes is not None and len(p.boxes) > 0:
#                         for plate_box in p.boxes:
#                             px1, py1, px2, py2 = map(int, plate_box.xyxy[0])
#                             py1 = max(0, py1)
#                             px1 = max(0, px1)
#                             py2 = min(roi.shape[0], py2)
#                             px2 = min(roi.shape[1], px2)
#                             plate_img = roi[py1:py2, px1:px2]

#                             if plate_img.size == 0:
#                                 continue

#                             result = ocr.ocr(plate_img, cls=True)

#                             if result and result[0]:
#                                 plate_text = result[0][0][1][0].strip()
#                                 print(f"车辆{plate_text}已进入")

#                                 if plate_text not in WHITELIST:
#                                     cv.putText(frame, f"陌生车辆: {plate_text}", (x1, y1-10),
#                                                cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
#                                     stranger_alert = True
#                                     with num_lock:
#                                         num += 1

                                  
#                                     save_snapshot(frame, f"Stranger_Car_{plate_text}", 0.9)
#                                 else:
#                                     cv.putText(frame, f"白名单: {plate_text}", (x1, y1-30),
#                                                cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

#                                 cv.rectangle(frame, (x1+px1, y1+py1), (x1+px2, y1+py2), (255, 255, 0), 2)
#     return frame, stranger_alert


def main(source):
    try:
        if isinstance(source, str) and source.isdigit():
            source = int(source)

        cap = cv.VideoCapture(source)
        if not cap.isOpened():
            raise Exception(f"无法打开视频源: {source}")
        print(f"视频源 {source} 开启成功")
    except Exception as e:
        print(f"视频源开启失败: {e}")
        error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv.putText(error_frame, "Video Error", (100, 240),
                  cv.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        ret, buffer = cv.imencode('.jpg', error_frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("视频读取完毕或出错")
            break

        frame = fire_recognize(frame)
        # frame = water_recognize(frame)
        # frame, stranger_alert = car_recognize(frame)
        frame = face_recognize(frame)

        ret, buffer = cv.imencode('.jpg', frame, [cv.IMWRITE_JPEG_QUALITY, 70])
        if not ret:
            continue
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()


app.secret_key = 'my_demo_key'
CORS(app, origins="*", methods=["GET", "POST", "OPTIONS"])

user = {
    'lwc': '123',
    'zjt': '123',
    'gdp': '123',
    'ljx': '123',
    'fyf': '123',
    'xjg': '123',
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(main(0), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/login', methods=['POST'])
def login():
    data = request.json
    username = data.get('username')
    password = data.get('password')

    if username in user and password == user[username]:
        session['username'] = username
        return jsonify({"success": True,'user':username, "message": "登录成功" }), 200
    else:
        return jsonify({"success": False, "message": "登录失败"}), 401

@app.route('/api/Register', methods=['POST'])
def register():
    data = request.json
    username = data.get('username')
    password = data.get('password')
    if username in user:
        return jsonify({'message': '用户名已存在'}), 400
    else:
        user[username] = password
        with open('user.txt', 'a') as f:
            f.write(username + '\n')
            f.write(password + '\n')
        return jsonify({'message': '注册成功'}), 200

@app.route('/api/user', methods=['GET'])
def get_user():
    username = session.get('username')
    if username:
        return jsonify({'user': username})
    else:
        return jsonify({'user': ''}), 401

@app.route('/api/video', methods=['GET'])
def video():
    video_path = request.args.get('path', '')
    if not video_path:
        return Response(main(0), mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return Response(main(video_path), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/num', methods=['GET'])
def get_num():
    with num_lock:
        return jsonify({'num': num})


@app.route('/api/snapshots', methods=['GET'])
def get_snapshots():
    """获取截图列表"""
    try:
        limit = request.args.get('limit', 20, type=int)
        conn, cursor = get_db_connection()

        cursor.execute('''
            SELECT a.id, a.timestamp, a.alert_type, a.confidence, i.file_path
            FROM alerts a
            JOIN alert_images i ON a.id = i.alert_id
            ORDER BY a.timestamp DESC
            LIMIT ?
        ''', (limit,))

        snapshots = []
        for row in cursor.fetchall():
            snapshots.append({
                'id': row[0],
                'timestamp': row[1],
                'type': row[2],
                'confidence': row[3],
                'file_path': row[4]
            })

        conn.close()
        return jsonify({'snapshots': snapshots})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/snapshots/<int:snapshot_id>', methods=['GET'])
def get_snapshot_image(snapshot_id):
    """获取截图图片"""
    try:
        conn, cursor = get_db_connection()
        cursor.execute('SELECT file_path FROM alert_images WHERE id = ?', (snapshot_id,))
        result = cursor.fetchone()
        conn.close()

        if result and result[0] and os.path.exists(result[0]):
            image = cv.imread(result[0])
            if image is not None:
                ret, buffer = cv.imencode('.jpg', image)
                if ret:
                    return Response(buffer.tobytes(), mimetype='image/jpeg')

        return "Image not found", 404
    except Exception as e:
        return str(e), 500

if __name__ == '__main__':
    os.makedirs('models', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    app.run(debug=True, host='0.0.0.0')
