import cv2
import mediapipe as mp
import time
import numpy as np

# --- AYARLAR ---
# Göz açıklık oranı (EAR) eşiği. Gözlükle genelde 0.20-0.25 arasıdır.
EAR_THRESHOLD = 0.22 
BLINK_MIN = 0.15      # Minimum göz kapama süresi (sn)
SEQ_TIMEOUT = 1.6     # Komut bekleme süresi

COMMANDS = {
    ("K",): "SU ISTIYORUM",
    ("K", "K"): "YARDIM CAGIRIN!",
    ("K", "K", "K"): "TESEKKUR EDERIM",
    ("U",): "EVET / ONAY",
    ("U", "U"): "HAYIR / RED",
    ("K", "U"): "AGRIM VAR"
}

# Mediapipe Çizim ve Yüz Mesh Ayarları
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)

def get_ear(landmarks, eye_indices):
    """Göz Açıklık Oranını (EAR) hesaplar."""
    # Dikey noktalar
    v1 = np.linalg.norm(np.array(landmarks[eye_indices[1]]) - np.array(landmarks[eye_indices[5]]))
    v2 = np.linalg.norm(np.array(landmarks[eye_indices[2]]) - np.array(landmarks[eye_indices[4]]))
    # Yatay nokta
    h = np.linalg.norm(np.array(landmarks[eye_indices[0]]) - np.array(landmarks[eye_indices[3]]))
    return (v1 + v2) / (2.0 * h)

# Gözlerin Mediapipe üzerindeki nokta indeksleri
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def main():
    cap = cv2.VideoCapture(0)
    eye_state = "ACIK"
    close_start = 0
    sequence = []
    last_action_time = time.time()
    display_msg = "SISTEM HAZIR"
    msg_expiry = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            mesh_coords = []
            for landmark in results.multi_face_landmarks[0].landmark:
                mesh_coords.append([landmark.x, landmark.y])
            
            # İki gözün ortalama açıklığına bakıyoruz
            ear_left = get_ear(mesh_coords, LEFT_EYE)
            ear_right = get_ear(mesh_coords, RIGHT_EYE)
            avg_ear = (ear_left + ear_right) / 2.0

            # Durum Tespiti
            curr = "KAPALI" if avg_ear < EAR_THRESHOLD else "ACIK"

            if curr == "KAPALI" and eye_state == "ACIK":
                close_start = time.time()
            elif curr == "ACIK" and eye_state == "KAPALI":
                dur = time.time() - close_start
                if dur > BLINK_MIN:
                    sequence.append("K" if dur < 0.5 else "U")
                    last_action_time = time.time()
            eye_state = curr

            # Görselleştirme (Göz noktalarını çizelim ki çalıştığını gör)
            h, w, _ = frame.shape
            for idx in LEFT_EYE + RIGHT_EYE:
                pt = results.multi_face_landmarks[0].landmark[idx]
                cv2.circle(frame, (int(pt.x * w), int(pt.y * h)), 1, (0, 255, 0), -1)
            
            color = (0, 0, 255) if eye_state == "KAPALI" else (255, 255, 0)
            cv2.putText(frame, f"EAR: {avg_ear:.2f}", (20, 80), 1, 1.5, color, 2)

        # Komut İşleme
        if sequence and (time.time() - last_action_time) > SEQ_TIMEOUT:
            cmd_key = tuple(sequence)
            display_msg = COMMANDS.get(cmd_key, f"TANIMSIZ: {'-'.join(sequence)}")
            msg_expiry = time.time() + 4.0
            sequence = []

        # Arayüz
        h, w, _ = frame.shape
        cv2.rectangle(frame, (0, h-60), (w, h), (20, 20, 20), -1)
        if time.time() < msg_expiry:
            cv2.putText(frame, display_msg, (20, h-20), 1, 1.8, (0, 255, 255), 2)
        cv2.putText(frame, f"DIZI: {'-'.join(sequence)}", (20, 40), 1, 1.5, (0, 255, 0), 2)

        cv2.imshow("Mediapipe Blink Comm", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()