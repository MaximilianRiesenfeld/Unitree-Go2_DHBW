"""
Go2 Live-Kamera mit Tür-/Fenstererkennung
-----------------------------------------
Zeigt das Frontkamera-Bild des Unitree Go2 live auf dem Laptop und zieht mit dem
eigenen YOLO-Modell (best.pt aus ~/unitree_ws/src/yolo_zeugs) Rahmen um Türen und Fenster:
    hellgrün = geschlossen      rot = offen

Modell-Klassen: 0 closed door, 1 closed window, 2 opened door, 3 opened window

Installation (auf dem Laptop):
    pip install "go2-webrtc-connect[video]" ultralytics opencv-python

Modell vom Pi holen (Tailscale muss laufen):
    scp fynn@100.121.161.1:~/unitree_ws/src/yolo_zeugs/best.pt .

Aufruf:
    python go2_tuer_fenster.py webcam                         # Test mit Laptop-Kamera, ohne Hund
    python go2_tuer_fenster.py ap                             # Laptop im WLAN des Hundes
    python go2_tuer_fenster.py lokal --ip 192.168.123.161     # gleiches Netzwerk
    python go2_tuer_fenster.py remote --sn B42D... --user mail --pw passwort   # anderes Netzwerk
    Optionen: --model best.pt  --conf 0.35  --frames 3

Tasten im Fenster:  q = beenden,  s = Screenshot speichern
Wichtig: Unitree-App vorher schließen (nur eine WebRTC-Verbindung gleichzeitig).
"""

import argparse
import asyncio
import logging
import queue
import threading
import time
from collections import defaultdict

import cv2
from ultralytics import YOLO

logging.basicConfig(level=logging.WARNING)

# Farben in BGR
HELLGRUEN = (144, 238, 144)
ROT = (0, 0, 255)

frame_queue: "queue.Queue" = queue.Queue(maxsize=2)
stop_event = threading.Event()


# --------------------------------------------------------------------------- Bildquellen
def push_frame(img):
    """Immer nur das neueste Bild behalten -> geringe Latenz."""
    if frame_queue.full():
        try:
            frame_queue.get_nowait()
        except queue.Empty:
            pass
    frame_queue.put(img)


def start_webcam(index: int = 0):
    def run():
        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            print("Webcam konnte nicht geöffnet werden.")
            stop_event.set()
            return
        while not stop_event.is_set():
            ok, img = cap.read()
            if ok:
                push_frame(img)
        cap.release()

    threading.Thread(target=run, daemon=True).start()


def start_go2(args):
    # Import erst hier, damit der Webcam-Test auch ohne go2-Paket läuft
    from go2_webrtc_driver.webrtc_driver import Go2WebRTCConnection, WebRTCConnectionMethod

    if args.modus == "ap":
        conn = Go2WebRTCConnection(WebRTCConnectionMethod.LocalAP)
    elif args.modus == "lokal":
        if args.ip:
            conn = Go2WebRTCConnection(WebRTCConnectionMethod.LocalSTA, ip=args.ip)
        elif args.sn:
            conn = Go2WebRTCConnection(WebRTCConnectionMethod.LocalSTA, serialNumber=args.sn)
        else:
            raise SystemExit("Modus 'lokal' braucht --ip oder --sn")
    else:  # remote
        if not (args.sn and args.user and args.pw):
            raise SystemExit("Modus 'remote' braucht --sn, --user und --pw (Unitree-Account)")
        conn = Go2WebRTCConnection(WebRTCConnectionMethod.Remote, serialNumber=args.sn,
                                   username=args.user, password=args.pw)

    async def recv_camera_stream(track):
        while not stop_event.is_set():
            frame = await track.recv()
            push_frame(frame.to_ndarray(format="bgr24"))

    loop = asyncio.new_event_loop()

    def run():
        asyncio.set_event_loop(loop)

        async def setup():
            try:
                await conn.connect()
                conn.video.switchVideoChannel(True)
                conn.video.add_track_callback(recv_camera_stream)
                print("Mit Go2 verbunden - warte auf Bild ...")
            except Exception as e:
                print(f"Verbindung fehlgeschlagen: {e}")
                stop_event.set()

        loop.run_until_complete(setup())
        loop.run_forever()

    threading.Thread(target=run, daemon=True).start()
    return loop


# --------------------------------------------------------------------------- Erkennung
def klasse_deuten(name: str):
    """'opened door' -> ('Tuer', True)"""
    n = name.lower()
    art = "Tuer" if "door" in n else "Fenster" if "window" in n else name
    offen = n.startswith("open")
    return art, offen


def zeichne_box(img, x1, y1, x2, y2, text, farbe):
    cv2.rectangle(img, (x1, y1), (x2, y2), farbe, 3)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    y_text = max(y1, th + 8)
    cv2.rectangle(img, (x1, y_text - th - 8), (x1 + tw + 8, y_text), farbe, -1)
    cv2.putText(img, text, (x1 + 4, y_text - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)


def waehle_device():
    try:
        import torch
        if torch.cuda.is_available():
            return 0
        if torch.backends.mps.is_available():  # Apple Silicon
            return "mps"
    except Exception:
        pass
    return "cpu"


# --------------------------------------------------------------------------- Hauptprogramm
def main():
    p = argparse.ArgumentParser(description="Go2 Kamera mit Tuer-/Fenstererkennung")
    p.add_argument("modus", choices=["webcam", "ap", "lokal", "remote"])
    p.add_argument("--ip")
    p.add_argument("--sn")
    p.add_argument("--user")
    p.add_argument("--pw")
    p.add_argument("--cam", type=int, default=0, help="Webcam-Index (Modus webcam)")
    p.add_argument("--model", default="best.pt", help="YOLO-Modell (best.pt oder best.onnx)")
    p.add_argument("--conf", type=float, default=0.35, help="Mindest-Konfidenz")
    p.add_argument("--frames", type=int, default=3,
                   help="Ein Objekt erst nach so vielen Bildern in Folge anzeigen (gegen Flackern)")
    args = p.parse_args()

    model = YOLO(args.model)
    device = waehle_device()
    print(f"Modell: {args.model}  Klassen: {model.names}  Device: {device}")

    loop = None
    if args.modus == "webcam":
        start_webcam(args.cam)
    else:
        loop = start_go2(args)

    fenster = "Go2 - Tueren & Fenster (q = Ende, s = Screenshot)"
    cv2.namedWindow(fenster, cv2.WINDOW_NORMAL)
    streak = defaultdict(int)  # Klasse -> Anzahl aufeinanderfolgender Bilder mit Treffer
    last, fps = time.time(), 0.0

    try:
        while not stop_event.is_set():
            try:
                img = frame_queue.get(timeout=0.5)
            except queue.Empty:
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            res = model.predict(img, conf=args.conf, device=device, verbose=False)[0]

            gesehen = set()
            boxen = []
            for b in res.boxes:
                cls = int(b.cls)
                gesehen.add(cls)
                boxen.append((cls, float(b.conf), *map(int, b.xyxy[0].tolist())))
            for cls in list(streak):
                if cls not in gesehen:
                    streak[cls] = 0
            for cls in gesehen:
                streak[cls] += 1

            n_offen = n_zu = 0
            for cls, conf, x1, y1, x2, y2 in boxen:
                if streak[cls] < args.frames:
                    continue
                art, offen = klasse_deuten(model.names[cls])
                farbe = ROT if offen else HELLGRUEN
                zustand = "offen" if offen else "zu"
                zeichne_box(img, x1, y1, x2, y2, f"{art} {zustand} {conf:.0%}", farbe)
                n_offen += offen
                n_zu += not offen

            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - last, 1e-6))
            last = now
            info = f"{fps:4.1f} FPS | offen: {n_offen} | zu: {n_zu}"
            cv2.putText(img, info, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
            cv2.putText(img, info, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow(fenster, img)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                name = time.strftime("go2_%Y%m%d_%H%M%S.jpg")
                cv2.imwrite(name, img)
                print(f"Gespeichert: {name}")
    finally:
        stop_event.set()
        cv2.destroyAllWindows()
        if loop is not None:
            loop.call_soon_threadsafe(loop.stop)


if __name__ == "__main__":
    main()
