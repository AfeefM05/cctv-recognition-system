import mysql.connector
from datetime import datetime

class FaceDBLoggerSQL:
    def __init__(self, host="localhost", port=3306, user="root", password="", database="face_db"):
        self.conn = mysql.connector.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database=database
        )
        self.cursor = self.conn.cursor()
        self.last_detection_times={}
        self.init_table()

    def init_table(self):
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS detections (
                id INT AUTO_INCREMENT PRIMARY KEY,
                roll_no VARCHAR(100),
                confidence FLOAT,
                timestamp DATETIME,
                camera INT
            )
        ''')
        self.conn.commit()

    def log_detection(self, name, confidence,camera):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        query = "INSERT INTO detections (roll_no, confidence, timestamp, camera) VALUES (%s, %s, %s, %s)"
        self.cursor.execute(query, (name, confidence, timestamp, camera))
        self.conn.commit()

    def should_log_detection(self, name):
        now = datetime.now()

        if name not in self.last_detection_times:
            self.last_detection_times[name] = now
            return True

        last_seen = self.last_detection_times[name]
        if (now - last_seen).total_seconds() >= 60:
            self.last_detection_times[name] = now
            return True

        return False

    def close(self):
        self.cursor.close()
        self.conn.close()
