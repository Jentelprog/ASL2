"""
ASL Predictor - Modern UI (PyQt5)
File: asl_modern_fixed_en.py

Dependencies:
    pip install PyQt5 opencv-python joblib cvzone

Usage:
    python asl_modern_fixed_en.py
"""

import sys
import os
import joblib
import cv2
from cvzone.HandTrackingModule import HandDetector

from PyQt5.QtCore import QTimer, Qt, QPropertyAnimation, QEasingCurve
from PyQt5.QtGui import QImage, QPixmap, QFont, QIcon, QColor
from PyQt5.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QFileDialog,
    QMessageBox,
    QFrame,
    QGraphicsDropShadowEffect,
    QGraphicsOpacityEffect,
    QShortcut,
)
from PyQt5.QtGui import QKeySequence


def hand2point(hand_roi, detector):
    ret = []
    hands, frame = detector.findHands(hand_roi, draw=True, flipType=True)
    if hands:
        hand1 = hands[0]
        point = hand1["lmList"]
        fingers1 = detector.fingersUp(hand1)
        for p in point:
            for i in p:
                ret.append(i)
        for f in fingers1:
            ret.append(f)
        return ret, frame
    return None


def predictsign(points, loaded_model):
    predictions = loaded_model.predict([points])
    return predictions


def apply_shadow(widget, blur=18, x_offset=0, y_offset=8, color=Qt.gray, alpha=0.15):
    shadow = QGraphicsDropShadowEffect(widget)
    shadow.setBlurRadius(blur)
    c = QColor(color)
    c.setAlphaF(alpha)
    shadow.setColor(c)
    shadow.setOffset(x_offset, y_offset)
    widget.setGraphicsEffect(shadow)


def apply_fade(widget, start=0.0, end=1.0, duration=600):
    old_effect = widget.graphicsEffect()
    if old_effect:
        widget.setGraphicsEffect(None)
    effect = QGraphicsOpacityEffect(widget)
    widget.setGraphicsEffect(effect)
    anim = QPropertyAnimation(effect, b"opacity", widget)
    anim.setStartValue(start)
    anim.setEndValue(end)
    anim.setDuration(duration)
    anim.setEasingCurve(QEasingCurve.InOutCubic)
    anim.start(QPropertyAnimation.DeleteWhenStopped)


class CardFrame(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("""
            QFrame {
                background: rgba(255,255,255,0.7);
                border: 1px solid rgba(245,245,240,0.8);
                border-radius: 12px;
            }
        """)


class ASLWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🤟 ASL Translator ")
        self.resize(1000, 720)

        self.setStyleSheet("""
        QMainWindow {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 #f8f4f0, stop:0.5 #f5f1eb, stop:1 #f0ebe5);
            color: #5a4c3e;
            font-family: 'Segoe UI', Verdana, Arial;
        }
        QLabel#titleLabel {
            color: #7a6a5b;
            font-weight: 700;
            font-size: 20px;
        }
        QLabel#subTitle {
            color: #8b7d6e;
            font-size: 12px;
        }
        QPushButton {
            cursor: pointer;
        }
        QPushButton.appButton {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 #d4c4b5, stop:1 #c8b8a8);
            color: #5a4c3e;
            border-radius: 10px;
            padding: 8px 14px;
            font-weight: 600;
            min-width: 100px;
            border: 1px solid #b8a897;
        }
        QPushButton.appButton:hover {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 #e0d5c8, stop:1 #d4c9bc);
        }
        QPushButton.ghost {
            background: transparent;
            color: #8b7d6e;
            border: 1px solid rgba(139,125,110,0.3);
            padding: 6px 12px;
            border-radius: 9px;
        }
        QLabel#videoLabel {
            border-radius: 14px;
            background: rgba(255,255,255,0.6);
            border: 1px solid rgba(245,245,240,0.8);
        }
        QLabel#roiLabel {
            border-radius: 10px;
            background: rgba(255,255,255,0.6);
            color: #8b7d6e;
            border: 1px solid rgba(245,245,240,0.8);
            font-size: 13px;
            font-weight: 600;
        }
        QLabel#sentenceLabel {
            font-size: 16px;
            color: #5a4c3e;
            padding: 8px;
            border-radius: 10px;
            background: rgba(255,255,255,0.8);
            border: 1px solid rgba(245,245,240,0.9);
        }
        QLabel#videoTitle, QLabel#roiTitle, QLabel#sentenceTitle {
            color: #7a6a5b;
            font-weight: 700;
            font-size: 15px;
        }
        """)

        # Load model
        model_path = os.path.join("models", "model_asl.pkl")
        if not os.path.exists(model_path):
            QMessageBox.critical(self, "Error", f"Model not found: {model_path}")
            raise FileNotFoundError(model_path)
        self.loaded_model = joblib.load(model_path)

        # Camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            QMessageBox.critical(self, "Error", "Unable to open camera.")
            sys.exit(1)

        self.detector = HandDetector(staticMode=False, maxHands=1, detectionCon=0.6, minTrackCon=0.5)

        # HEADER
        header = QWidget()
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(16, 10, 16, 10)
        title = QLabel("🤟 ASL Translator")
        title.setObjectName("titleLabel")
        subtitle = QLabel("Realtime sign recognition")
        subtitle.setObjectName("subTitle")
        left_header = QVBoxLayout()
        left_header.addWidget(title)
        left_header.addWidget(subtitle)
        header_layout.addLayout(left_header)
        header_layout.addStretch()
        help_btn = QPushButton("❔ Help")
        help_btn.setProperty("class", "ghost")
        help_btn.clicked.connect(self.show_help)
        header_layout.addWidget(help_btn)
        apply_shadow(header, blur=4, y_offset=2, color=Qt.gray)

        # MAIN
        main_container = QWidget()
        main_layout = QHBoxLayout(main_container)
        main_layout.setContentsMargins(18, 8, 18, 18)
        main_layout.setSpacing(18)

        # Left: Camera
        left_card = CardFrame()
        left_layout = QVBoxLayout(left_card)
        video_title = QLabel("📷 Camera View")
        video_title.setObjectName("videoTitle")
        self.video_label = QLabel()
        self.video_label.setObjectName("videoLabel")
        self.video_label.setFixedSize(640, 480)
        self.video_label.setAlignment(Qt.AlignCenter)
        apply_shadow(self.video_label, blur=22, y_offset=18, color=Qt.gray)
        left_layout.addWidget(video_title)
        left_layout.addWidget(self.video_label, alignment=Qt.AlignCenter)

        # Buttons
        btn_row = QWidget()
        btn_layout = QHBoxLayout(btn_row)
        btn_layout.setSpacing(10)
        self.btn_capture = QPushButton("📸 Capture (s)")
        self.btn_capture.setProperty("class", "appButton")
        self.btn_erase = QPushButton("⌫ Delete (d)")
        self.btn_erase.setProperty("class", "appButton")
        self.btn_space = QPushButton("␣ Space")
        self.btn_space.setProperty("class", "appButton")
        self.btn_save = QPushButton("💾 Save")
        self.btn_save.setProperty("class", "appButton")
        self.btn_clear = QPushButton("🗑️ Clear")
        self.btn_clear.setProperty("class", "ghost")
        self.btn_quit = QPushButton("🚪 Quit (q)")
        self.btn_quit.setProperty("class", "ghost")
        for b in [self.btn_capture, self.btn_erase, self.btn_space, self.btn_save, self.btn_clear, self.btn_quit]:
            b.setMinimumHeight(36)
        btn_layout.addWidget(self.btn_capture)
        btn_layout.addWidget(self.btn_erase)
        btn_layout.addWidget(self.btn_space)
        btn_layout.addWidget(self.btn_save)
        btn_layout.addWidget(self.btn_clear)
        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_quit)
        left_layout.addWidget(btn_row)

        # Right: ROI + sentence
        right_card = CardFrame()
        right_layout = QVBoxLayout(right_card)
        roi_title = QLabel("✋ Detected Hand")
        roi_title.setObjectName("roiTitle")
        self.roi_label = QLabel("Hand area\nwill appear here")
        self.roi_label.setObjectName("roiLabel")
        self.roi_label.setFixedSize(240, 240)
        self.roi_label.setAlignment(Qt.AlignCenter)
        sentence_title = QLabel("📝 Constructed Sentence")
        sentence_title.setObjectName("sentenceTitle")
        self.sentence_label = QLabel("Your sentence will appear here...")
        self.sentence_label.setObjectName("sentenceLabel")
        self.sentence_label.setWordWrap(True)
        quick_info = QLabel("• Real-time prediction • Shortcuts: s, d, space, q")
        quick_info.setStyleSheet("color:#9c8e7f; font-size:11px;")
        right_layout.addWidget(roi_title)
        right_layout.addWidget(self.roi_label, alignment=Qt.AlignCenter)
        right_layout.addWidget(sentence_title)
        right_layout.addWidget(self.sentence_label)
        right_layout.addWidget(quick_info)
        right_layout.addStretch()

        # Assembly
        main_layout.addWidget(left_card, stretch=3)
        main_layout.addWidget(right_card, stretch=1)

        # Central
        central = QWidget()
        c_layout = QVBoxLayout(central)
        c_layout.addWidget(header)
        c_layout.addWidget(main_container)
        self.setCentralWidget(central)

        # Variables
        self.sentence = ""
        self.current_frame = None
        self.current_hand_bbox = None

        # Connections
        self.btn_capture.clicked.connect(self.on_capture)
        self.btn_erase.clicked.connect(self.on_erase)
        self.btn_space.clicked.connect(self.on_space)
        self.btn_save.clicked.connect(self.on_save)
        self.btn_clear.clicked.connect(self.on_clear)
        self.btn_quit.clicked.connect(self.close)

        # Keyboard shortcuts
        QShortcut(QKeySequence("s"), self, activated=self.on_capture)
        QShortcut(QKeySequence("d"), self, activated=self.on_erase)
        QShortcut(QKeySequence("space"), self, activated=self.on_space)
        QShortcut(QKeySequence("q"), self, activated=self.close)

        # Timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

        # Effects
        apply_fade(self.video_label, 0, 1, 800)
        apply_fade(self.roi_label, 0, 1, 900)
        apply_fade(self.sentence_label, 0, 1, 1000)

    def show_help(self):
        text = (
            "Quick User Guide:\n\n"
            "- Bring your hand close to the camera. Click 📸 or press 's' to capture.\n"
            "- ⌫ deletes the last letter (or press 'd').\n"
            "- ␣ adds a space (or press 'space').\n"
            "- 🗑️ Clear empties the sentence.\n"
            "- 💾 Save to store the sentence in a .txt file.\n"
            "- If the hand is not detected, move it closer or improve lighting.\n\n"
            "UX Tip: natural light + plain background improves detection."
        )
        QMessageBox.information(self, "Help — ASL Translator", text)

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return
        self.current_frame = frame.copy()
        hands, annotated = self.detector.findHands(frame, draw=True, flipType=True)
        self.current_hand_bbox = hands[0].get("bbox", None) if hands else None
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        h, w, ch = annotated_rgb.shape
        qt_img = QImage(annotated_rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qt_img).scaled(
            self.video_label.width(), self.video_label.height(), Qt.KeepAspectRatio
        )
        self.video_label.setPixmap(pix)

    def on_capture(self):
        if self.current_frame is None or self.current_hand_bbox is None:
            QMessageBox.information(self, "Info", "Hand not detected.")
            return
        x, y, w, h = self.current_hand_bbox
        height, width = self.current_frame.shape[:2]
        pad = 60
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(width, x + w + pad), min(height, y + h + pad)
        hand_roi = self.current_frame[y1:y2, x1:x2]
        if hand_roi.size == 0:
            QMessageBox.warning(self, "Warning", "Empty ROI.")
            return
        roi_rgb = cv2.cvtColor(hand_roi, cv2.COLOR_BGR2RGB)
        qt_roi = QImage(roi_rgb.data, roi_rgb.shape[1], roi_rgb.shape[0],
                        roi_rgb.shape[1]*3, QImage.Format_RGB888)
        pix_roi = QPixmap.fromImage(qt_roi).scaled(self.roi_label.width(),
                        self.roi_label.height(), Qt.KeepAspectRatio)
        self.roi_label.setPixmap(pix_roi)
        result = hand2point(hand_roi, self.detector)
        if not result:
            QMessageBox.information(self, "Info", "Unclear hand. Try again.")
            return
        points, _ = result
        try:
            p = predictsign(points=points, loaded_model=self.loaded_model)
        except Exception as e:
            QMessageBox.critical(self, "Prediction Error", str(e))
            return
        self.sentence += str(p[0])
        self.sentence_label.setText(self.sentence)
        apply_fade(self.sentence_label, 0.4, 1.0, 200)

    def on_erase(self):
        self.sentence = self.sentence[:-1]
        self.sentence_label.setText(self.sentence or "Your sentence will appear here...")

    def on_space(self):
        self.sentence += " "
        self.sentence_label.setText(self.sentence)

    def on_clear(self):
        self.sentence = ""
        self.sentence_label.setText("Your sentence will appear here...")

    def on_save(self):
        if not self.sentence:
            QMessageBox.information(self, "Info", "Nothing to save.")
            return
        fname, _ = QFileDialog.getSaveFileName(self, "Save", "", "Text Files (*.txt)")
        if fname:
            with open(fname, "w", encoding="utf-8") as f:
                f.write(self.sentence)
            QMessageBox.information(self, "Success", f"Saved in {fname}")

    def closeEvent(self, event):
        self.timer.stop()
        if self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 10))
    window = ASLWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
