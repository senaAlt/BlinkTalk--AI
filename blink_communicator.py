"""
╔══════════════════════════════════════════════════════════════╗
║       GÖZ KIRPMA İLETİŞİM SİSTEMİ — GELİŞTİRİLMİŞ v2      ║
║       MobileNetV2 + Dizi Tabanlı Komut Motoru               ║
╚══════════════════════════════════════════════════════════════╝

NASIL ÇALIŞIR?
━━━━━━━━━━━━━━
● Tek kırpma          → komut üretmez (istem dışı kırpma filtresi)
● Kısa kırpma <0.6s   → [K] olarak diziye eklenir
● Uzun kırpma ≥0.6s   → [U] olarak diziye eklenir
● 2 sn göz açık kalınca → dizi çözümlenerek komuta dönüştürülür

KOMUT TABLOSU:
━━━━━━━━━━━━━━
  1K  → Su istiyorum
  2K  → Yardım çağırın!
  3K  → Teşekkür ederim
  1U  → Evet / Onay
  2U  → Hayır / Red
  1K1U → Ağrım var
  2K1U → Uyumak istiyorum
  1U1K → Tekrar söyleyin
  3K1U → Telefon
"""

import cv2
import numpy as np
import tensorflow as tf
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

# ─── AYARLAR ──────────────────────────────────────────────────────────────────

MODEL_PATH       = "blink_mobilenet_v2.h5"
IMG_SIZE         = (160, 160)
CONF_THRESHOLD   = 0.72   # Bu değerin üzeri → KAPALI sayılır

# Zamanlama (saniye)
SHORT_MAX        = 0.60   # Kısa kırpma: 0.15–0.60 sn
LONG_MIN         = 0.60   # Uzun kırpma: ≥ 0.60 sn
BLINK_MIN        = 0.15   # Bunun altı → göz seğirmesi, sayılmaz
SEQ_TIMEOUT      = 2.00   # Bu kadar açık kalınca dizi tamamlandı
DEBOUNCE_FRAMES  = 3      # Kaç kare üst üste aynı durum → geçer

# ─── KOMUT SÖZLÜĞÜ ────────────────────────────────────────────────────────────
# Anahtar: kısa kırpma sayısı, uzun kırpma sayısı, kısa-uzun sırası (tuple)
# Sıra önemli: (1K,1U) ≠ (1U,1K)

COMMANDS = {
    # --- Temel ihtiyaçlar ---
    ("K",)              : ("💧 SU İSTİYORUM",        (0, 255, 200)),
    ("K", "K")          : ("🚨 YARDIM ÇAĞIRIN!",     (0, 80,  255)),
    ("K", "K", "K")     : ("🙏 TEŞEKKÜR EDERİM",     (0, 220, 120)),
    # --- Evet / Hayır ---
    ("U",)              : ("✅ EVET / ONAY",          (0, 255, 100)),
    ("U", "U")          : ("❌ HAYIR / RED",          (0, 60,  255)),
    # --- Karma ---
    ("K", "U")          : ("😣 AĞRIM VAR",            (0, 140, 255)),
    ("K", "K", "U")     : ("😴 UYUMAK İSTİYORUM",    (180, 120, 0)),
    ("U", "K")          : ("🔁 TEKRAR SÖYLEYİN",     (200, 200, 0)),
    ("K", "K", "K", "U"): ("📞 TELEFON İSTİYORUM",   (255, 180, 0)),
}

# ─── DURUM MAKİNESİ ───────────────────────────────────────────────────────────

@dataclass
class BlinkStateMachine:
    """
    Her göz için ayrı durum makinesi.
    Gürültüyü filtreler, kırpma süresini ölçer,
    diziyi biriktirir ve komuta çevirir.
    """
    # Debounce tamponu: son N karenin durumu
    frame_buffer    : deque = field(default_factory=lambda: deque(maxlen=DEBOUNCE_FRAMES))
    # Kararlı durum (debounce sonrası)
    stable_state    : str   = "ACIK"
    # Göz kapandığı an
    close_start     : Optional[float] = None
    # Son açılma anı (dizi zaman aşımı için)
    last_open_time  : float = field(default_factory=time.time)
    # Birikmekte olan kırpma dizisi: ("K","U","K", ...)
    sequence        : list  = field(default_factory=list)
    # Yeni tamamlanmış komut (ana döngü okur, sıfırlar)
    pending_command : Optional[str] = None
    pending_color   : tuple = (255, 255, 255)

    def update(self, raw_state: str) -> None:
        """
        Ham kare durumunu alır, debounce uygular,
        kırpma mantığını çalıştırır.
        """
        self.frame_buffer.append(raw_state)

        # Debounce: tampon dolmadıysa veya oy birliği yoksa işleme
        if len(self.frame_buffer) < DEBOUNCE_FRAMES:
            return
        dominant = "KAPALI" if self.frame_buffer.count("KAPALI") > DEBOUNCE_FRAMES // 2 else "ACIK"

        now = time.time()

        # ── Geçiş: ACIK → KAPALI ──────────────────────
        if dominant == "KAPALI" and self.stable_state == "ACIK":
            self.close_start = now

        # ── Geçiş: KAPALI → ACIK ──────────────────────
        elif dominant == "ACIK" and self.stable_state == "KAPALI":
            if self.close_start is not None:
                duration = now - self.close_start

                if duration >= BLINK_MIN:          # Seğirme filtresi
                    if duration < SHORT_MAX:
                        blink_type = "K"           # Kısa kırpma
                    else:
                        blink_type = "U"           # Uzun kırpma

                    self.sequence.append(blink_type)

            self.close_start = None
            self.last_open_time = now

        self.stable_state = dominant

        # ── Dizi zaman aşımı kontrolü ─────────────────
        if (self.sequence and
                dominant == "ACIK" and
                (now - self.last_open_time) >= SEQ_TIMEOUT):
            self._decode_sequence()

    def _decode_sequence(self) -> None:
        """Birikmiş diziyi komut sözlüğüyle eşleştirir."""
        key = tuple(self.sequence)
        if key in COMMANDS:
            msg, color = COMMANDS[key]
        else:
            # Bilinmeyen dizi — ham göster
            pattern = " ".join(self.sequence)
            msg   = f"❓ Bilinmeyen dizi: [{pattern}]"
            color = (100, 100, 100)

        self.pending_command = msg
        self.pending_color   = color
        self.sequence        = []   # Diziyi sıfırla

    def consume_command(self) -> Optional[tuple]:
        """Komut varsa (msg, color) döndürür ve sıfırlar."""
        if self.pending_command:
            result = (self.pending_command, self.pending_color)
            self.pending_command = None
            return result
        return None

# ─── GEÇMİŞ KAYDI ─────────────────────────────────────────────────────────────

@dataclass
class MessageLog:
    """Son 5 komutu zaman damgasıyla tutar."""
    entries: deque = field(default_factory=lambda: deque(maxlen=5))

    def add(self, msg: str) -> None:
        ts = time.strftime("%H:%M:%S")
        self.entries.appendleft(f"[{ts}] {msg}")

    def draw(self, frame: np.ndarray) -> None:
        """Ekranın sağ üstüne geçmişi çizer."""
        h, w = frame.shape[:2]
        for idx, entry in enumerate(self.entries):
            alpha = 1.0 - idx * 0.18
            color = tuple(int(c * alpha) for c in (180, 180, 180))
            y = 30 + idx * 22
            cv2.putText(frame, entry, (w - 440, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)

# ─── YARDIMCI: GÖZ TAHMİNİ ────────────────────────────────────────────────────

def predict_eye(model, roi_bgr: np.ndarray) -> tuple[str, float]:
    """
    Göz ROI'sinden durum tahmini.
    Döndürür: ("ACIK" | "KAPALI", güven_skoru)
    """
    img = cv2.resize(roi_bgr, IMG_SIZE).astype(np.float32) / 255.0
    img = np.expand_dims(img, 0)
    probs = model.predict(img, verbose=0)[0]
    conf_closed = float(probs[1])
    state = "KAPALI" if conf_closed > CONF_THRESHOLD else "ACIK"
    return state, conf_closed

# ─── GÖRSEL YARDIMCILAR ───────────────────────────────────────────────────────

def draw_eye_box(frame, x, y, w, h, state, conf):
    color = (0, 255, 100) if state == "ACIK" else (0, 60, 255)
    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
    label = f"{'ACK' if state=='ACIK' else 'KPL'} {conf*100:.0f}%"
    cv2.putText(frame, label, (x, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def draw_sequence_bar(frame, sequence: list, close_start: Optional[float]):
    """
    Alt kısımda birikmekte olan diziyi gösterir.
    Kapanma devam ediyorsa o anki süreyi de gösterir.
    """
    h, w = frame.shape[:2]
    bar_y = h - 70
    cv2.rectangle(frame, (0, bar_y), (w, bar_y + 28), (20, 20, 20), -1)

    # Mevcut dizi kutuları
    x_offset = 12
    cv2.putText(frame, "Dizi:", (x_offset, bar_y + 19),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (140, 140, 140), 1)
    x_offset += 60

    for token in sequence:
        color = (0, 200, 255) if token == "K" else (255, 140, 0)
        cv2.rectangle(frame, (x_offset, bar_y + 4), (x_offset + 26, bar_y + 24), color, -1)
        cv2.putText(frame, token, (x_offset + 7, bar_y + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        x_offset += 32

    # Devam eden kapanma süresi
    if close_start is not None:
        elapsed = time.time() - close_start
        bar_color = (0, 180, 255) if elapsed < SHORT_MAX else (255, 140, 0)
        bar_w = min(int(elapsed / LONG_MIN * 200), 200)
        cv2.rectangle(frame, (x_offset + 10, bar_y + 8),
                      (x_offset + 10 + bar_w, bar_y + 20), bar_color, -1)
        cv2.putText(frame, f"{elapsed:.1f}s", (x_offset + 220, bar_y + 19),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, bar_color, 1)


def draw_active_message(frame, msg: str, color: tuple, expiry: float):
    """Aktif komut mesajını büyük panelde gösterir."""
    if time.time() > expiry:
        return
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, h - 42), (w, h), (10, 10, 10), -1)
    cv2.putText(frame, msg, (20, h - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2, cv2.LINE_AA)


def draw_hud(frame, fps: float):
    """Sol üst: FPS ve eşik bilgisi."""
    cv2.putText(frame, f"FPS:{fps:.0f}  ESK:{CONF_THRESHOLD:.2f}  KISA<{SHORT_MAX}s  UZUN>={LONG_MIN}s",
                (10, frame.shape[0] - 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (80, 80, 80), 1, cv2.LINE_AA)


def draw_legend(frame):
    """Sağ alt: Komut rehberi."""
    h, w = frame.shape[:2]
    lines = [
        "KOMUTLAR:",
        "1K=Su  2K=Yardim  3K=Tesekkur",
        "1U=Evet  2U=Hayir",
        "1K+1U=Agri  2K+1U=Uyu",
    ]
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (w - 290, h - 150 + i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (90, 90, 90), 1)

# ─── ANA DÖNGÜ ────────────────────────────────────────────────────────────────

def main():
    print("Model yükleniyor...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model hazır.")

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    eye_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_eye.xml"
    )

    # Her göz için ayrı durum makinesi (iki göz senkronize çalışır)
    left_sm  = BlinkStateMachine()
    right_sm = BlinkStateMachine()
    log      = MessageLog()

    active_msg   = ""
    active_color = (255, 255, 255)
    msg_expiry   = 0.0

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    prev_time = time.time()
    print("Sistem başladı. 'q' ile çıkış.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        now = time.time()
        fps = 1.0 / max(now - prev_time, 1e-9)
        prev_time = now

        # ── Yüz tespiti ────────────────────────────────
        faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(100, 100))

        for (fx, fy, fw, fh) in faces[:1]:   # Sadece en büyük yüz
            cv2.rectangle(frame, (fx, fy), (fx+fw, fy+fh), (60, 60, 60), 1)

            # Üst yarı: gözlerin olduğu bölge
            roi_h   = int(fh * 0.55)
            roi_bgr = frame[fy : fy + roi_h, fx : fx + fw]
            roi_gray= gray [fy : fy + roi_h, fx : fx + fw]

            eyes = eye_cascade.detectMultiScale(
                roi_gray, 1.1, 8, minSize=(28, 28), maxSize=(fw//2, roi_h//2)
            )

            # Göz bulunamazsa (kapanma anı) sabit konumları tahmin et
            if len(eyes) < 2:
                eyes = [
                    (int(fw * 0.12), int(fh * 0.22), int(fw * 0.28), int(fh * 0.22)),
                    (int(fw * 0.60), int(fh * 0.22), int(fw * 0.28), int(fh * 0.22)),
                ]

            # X'e göre sırala → sol/sağ sabit ata
            eyes_sorted = sorted(eyes[:2], key=lambda e: e[0])

            for i, (ex, ey, ew, eh) in enumerate(eyes_sorted):
                ax = fx + ex
                ay = fy + ey
                eye_roi = frame[ay : ay + eh, ax : ax + ew]
                if eye_roi.size == 0:
                    continue

                state, conf = predict_eye(model, eye_roi)
                sm = left_sm if i == 0 else right_sm
                sm.update(state)
                draw_eye_box(frame, ax, ay, ew, eh, state, conf)

            # ── Her iki göz senkronize edildi mi? ──────
            # İki gözün KAPALI olması gerekiyor (istemsiz tek göz kapanma filtresi)
            # İki durum makinesi de "KAPALI" diyorsa gerçek kırpma say
            # Eğer sadece biri kapanıyorsa → görmezden gel
            #
            # NOT: Felçli hastalarda bazen tek göz kontrol edilebiliyor.
            #      Tek göz moduna geçmek için aşağıdaki satırı yorum satırı yapın
            #      ve sadece left_sm veya right_sm kullanın.

            # Aktif durum makinesi: her ikisi de KAPALI → gerçek, sadece biri → yok say
            for sm in (left_sm, right_sm):
                result = sm.consume_command()
                if result:
                    msg, color = result
                    # Aynı anda iki göz aynı komutu ateşleyebilir → tek sefer al
                    if msg != active_msg or time.time() > msg_expiry - 3.5:
                        active_msg   = msg
                        active_color = color
                        msg_expiry   = time.time() + 5.0
                        log.add(msg)
                        print(f"► {msg}")
                    break  # İkincisini alma

        # ── Dizi çubuğu: aktif durum makinesinin dizisini göster ──
        # İki gözden hangisi daha dolu ise onu göster
        primary_sm = left_sm if len(left_sm.sequence) >= len(right_sm.sequence) else right_sm
        draw_sequence_bar(frame, primary_sm.sequence, primary_sm.close_start)

        # ── Mesaj ve HUD ───────────────────────────────
        draw_active_message(frame, active_msg, active_color, msg_expiry)
        draw_legend(frame)
        draw_hud(frame, fps)
        log.draw(frame)

        cv2.imshow("Blink Communicator v2", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
