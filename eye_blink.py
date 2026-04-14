import cv2
import numpy as np
import tensorflow as tf
import time
from threading import Thread

# --- 1. AYARLAR ---
MODEL_PATH = "blink_mobilenet_v2.h5"
IMG_SIZE = (160, 160)    
CONF_THRESHOLD_HI = 0.85 
CONF_THRESHOLD_LO = 0.20 
BLINK_MIN = 0.18         
LONG_BLINK_THRESHOLD = 0.65 
SEQ_TIMEOUT = 1.8        

COMMANDS = {
    ("K",): "SU ISTIYORUM",
    ("K", "K"): "YARDIM CAGIRIN!",
    ("K", "K", "K"): "TESEKKUR EDERIM",
    ("U",): "EVET / ONAY",
    ("U", "U"): "HAYIR / RED",
    ("K", "U"): "AGRIM VAR"
}

# --- YENİ: EKRAN REHBERİ ---
GUIDE = [
    "KOMUT REHBERI",
    "K     : SU",
    "K-K   : YARDIM",
    "K-K-K : TESEKKUR",
    "U     : EVET",
    "U-U   : HAYIR",
    "K-U   : AGRI"
]

class EyeDetector:
    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path)
        self.prediction = 0.0
        self.running = True
        self.img_to_predict = None

    def start(self):
        Thread(target=self._predict_loop, daemon=True).start()

    def _predict_loop(self):
        while self.running:
            if self.img_to_predict is not None:
                res = self.model.predict(self.img_to_predict, verbose=0)
                self.prediction = res[0][1]
                self.img_to_predict = None
            else:
                time.sleep(0.01)

def main():
    detector = EyeDetector(MODEL_PATH)
    detector.start()
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    cap = cv2.VideoCapture(0)
    
    eye_state = "ACIK"
    close_start = 0
    sequence = []
    last_action_time = time.time()
    display_msg = "SISTEM HAZIR"
    msg_expiry = time.time()
    last_blink_info = "BEKLENIYOR..."

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w_f, h_f) in faces[:1]:
            roi_y, roi_h = y + int(h_f * 0.28), int(h_f * 0.22)
            roi_x, roi_w = x + int(w_f * 0.1), int(w_f * 0.8)
            eye_region = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
            
            if eye_region.size > 0:
                img = cv2.resize(eye_region, IMG_SIZE).astype(np.float32) / 255.0
                detector.img_to_predict = np.expand_dims(img, 0)
                
                prob = detector.prediction
                
                if prob > CONF_THRESHOLD_HI: curr = "KAPALI"
                elif prob < CONF_THRESHOLD_LO: curr = "ACIK"
                else: curr = eye_state

                if curr == "KAPALI" and eye_state == "ACIK":
                    close_start = time.time()
                elif curr == "ACIK" and eye_state == "KAPALI":
                    dur = time.time() - close_start
                    if dur > BLINK_MIN:
                        type_label = "K" if dur < LONG_BLINK_THRESHOLD else "U"
                        sequence.append(type_label)
                        last_blink_info = f"SON: {'KISA' if type_label=='K' else 'UZUN'} ({dur:.2f}s)"
                        last_action_time = time.time()
                
                eye_state = curr

                color = (0, 0, 255) if eye_state == "KAPALI" else (0, 255, 0)
                cv2.rectangle(frame, (roi_x, roi_y), (roi_x+roi_w, roi_y+roi_h), color, 2)

                if eye_state == "KAPALI":
                    current_dur = time.time() - close_start
                    bar_w = int(min(current_dur / 1.2, 1.0) * roi_w)
                    bar_color = (0, 255, 255) if current_dur < LONG_BLINK_THRESHOLD else (0, 140, 255)
                    
                    cv2.rectangle(frame, (roi_x, roi_y-20), (roi_x + bar_w, roi_y-5), bar_color, -1)
                    cv2.putText(frame, f"{current_dur:.2f}s", (roi_x + roi_w + 5, roi_y + 15), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, bar_color, 2)
                    
                    if current_dur >= LONG_BLINK_THRESHOLD:
                        cv2.putText(frame, "UZUN MOD", (roi_x, roi_y-30), 1, 1, (0, 165, 255), 2)

        if sequence and (time.time() - last_action_time) > SEQ_TIMEOUT:
            cmd_key = tuple(sequence)
            display_msg = COMMANDS.get(cmd_key, f"TANIMSIZ: {'-'.join(sequence)}")
            msg_expiry = time.time() + 5.0
            sequence = []

        # --- ARAYÜZ ÇİZİMLERİ ---
        
        # 1. SAĞ TARAF: KOMUT REHBERİ (YENİ)
        # Rehberin arka planını hafif karartalım
        overlay = frame.copy()
        cv2.rectangle(overlay, (w - 210, 10), (w - 10, 200), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        
        for i, line in enumerate(GUIDE):
            color = (255, 255, 255) if i == 0 else (200, 200, 200)
            scale = 0.6 if i == 0 else 0.45
            cv2.putText(frame, line, (w - 200, 35 + (i * 25)), 
                        cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1 if i != 0 else 2)

        # 2. SOL ÜST: Dizi Bilgisi
        cv2.putText(frame, f"DIZI: {'-'.join(sequence)}", (20, 40), 1, 2, (255, 255, 0), 2)
        cv2.putText(frame, last_blink_info, (20, 80), 1, 1.2, (200, 200, 200), 1)

        # 3. ALT PANEL: İstek Mesajı
        cv2.rectangle(frame, (0, h-60), (w, h), (25, 25, 25), -1)
        if time.time() < msg_expiry:
            cv2.putText(frame, f"ISTEK: {display_msg}", (20, h-20), 1, 2, (0, 255, 255), 2)

        cv2.imshow("Blink Communicator v5.1", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    detector.running = False
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()