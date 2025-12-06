import sys
import fitz  # PyMuPDF
import cv2
import mediapipe as mp
import math
import time

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QFileDialog, QToolBar, QAction,
    QStatusBar, QWidget, QVBoxLayout, QPushButton, QGroupBox, QDesktopWidget
)
from PyQt5.QtGui import QPixmap, QImage, QIcon
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer


# 제스처 인식 쓰레드
class HandGestureThread(QThread):
    change_pixmap_signal = pyqtSignal(QImage)
    gesture_signal = pyqtSignal(list, str)

    def __init__(self):
        super().__init__()
        self._run_flag = True
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils

    def get_finger_status(self, landmarks):
        tips = [4, 8, 12, 16, 20]
        joints = [3, 6, 10, 14, 18]
        wrist = landmarks[0]
        fingers = []
        for tip_idx, joint_idx in zip(tips, joints):
            tip = landmarks[tip_idx]
            joint = landmarks[joint_idx]
            dist_tip = math.sqrt((tip.x - wrist.x) ** 2 + (tip.y - wrist.y) ** 2)
            dist_joint = math.sqrt((joint.x - wrist.x) ** 2 + (joint.y - wrist.y) ** 2)
            fingers.append(1 if dist_tip > dist_joint else 0)
        return fingers

    def run(self):
        cap = cv2.VideoCapture(0)
        while self._run_flag:
            ret, cv_img = cap.read()
            if not ret: continue

            cv_img = cv2.flip(cv_img, 1)
            image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = self.hands.process(image)
            image.flags.writeable = True

            current_fingers = []
            hand_label = "None"

            if results.multi_hand_landmarks:
                for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    self.mp_drawing.draw_landmarks(image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    if results.multi_handedness:
                        hand_label = results.multi_handedness[i].classification[0].label
                    current_fingers = self.get_finger_status(hand_landmarks.landmark)

                    info_text = f"{hand_label}: {current_fingers}"
                    cv2.putText(image, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            self.gesture_signal.emit(current_fingers, hand_label)

            h, w, ch = image.shape
            bytes_per_line = ch * w
            qt_img = QImage(image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            p = qt_img.scaled(320, 240, Qt.KeepAspectRatio)
            self.change_pixmap_signal.emit(p)
            time.sleep(0.01)
        cap.release()

    def stop(self):
        self._run_flag = False
        self.wait()

# 컨트롤러 윈도우 (웹캠 + 설정) - 별도 창
class ControllerWindow(QWidget):
    # 메인 윈도우로 보낼 신호들
    req_next_page = pyqtSignal()
    req_prev_page = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setWindowTitle("발표자 컨트롤러 (Remote)")
        self.resize(350, 600)
        self.move(100, 100)  # 화면 왼쪽 상단에 위치

        # 로직 변수
        self.video_thread = None
        self.current_fingers = []
        self.current_hand_label = "None"
        self.recording_target = None
        self.saved_data_next = (None, None)
        self.saved_data_prev = (None, None)
        self.can_trigger = True

        # 쿨다운 타이머
        self.cooldown_timer = QTimer()
        self.cooldown_timer.setInterval(1500)
        self.cooldown_timer.setSingleShot(True)
        self.cooldown_timer.timeout.connect(self.reset_cooldown)

        self.init_ui()
        self.start_video_thread()

    def init_ui(self):
        layout = QVBoxLayout()

        # 웹캠 뷰
        self.webcam_label = QLabel("Webcam Loading...")
        self.webcam_label.setFixedSize(320, 240)
        self.webcam_label.setStyleSheet("background-color: #222; border: 1px solid #555; color: white;")
        self.webcam_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.webcam_label)

        # 상태 메시지
        self.status_label = QLabel("준비됨")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("font-size: 14px; font-weight: bold; margin: 10px 0;")
        layout.addWidget(self.status_label)

        # 설정 그룹
        grp = QGroupBox("제스처 설정")
        grp_layout = QVBoxLayout()

        self.btn_next = QPushButton("▶ 다음 페이지 설정")
        self.btn_next.clicked.connect(lambda: self.start_recording('next'))
        self.lbl_next = QLabel("설정 안됨")
        self.lbl_next.setAlignment(Qt.AlignCenter)
        self.lbl_next.setStyleSheet("color: gray;")

        self.btn_prev = QPushButton("◀ 이전 페이지 설정")
        self.btn_prev.clicked.connect(lambda: self.start_recording('prev'))
        self.lbl_prev = QLabel("설정 안됨")
        self.lbl_prev.setAlignment(Qt.AlignCenter)
        self.lbl_prev.setStyleSheet("color: gray;")

        grp_layout.addWidget(self.btn_next)
        grp_layout.addWidget(self.lbl_next)
        grp_layout.addWidget(self.btn_prev)
        grp_layout.addWidget(self.lbl_prev)
        grp.setLayout(grp_layout)
        layout.addWidget(grp)

        # 팁
        tip = QLabel("이 창은 발표자만 보는 화면입니다.\n메인 화면은 발표할 화면으로 옮기세요.")
        tip.setWordWrap(True)
        tip.setStyleSheet("color: #666; font-size: 12px;")
        layout.addWidget(tip)

        layout.addStretch()
        self.setLayout(layout)

    def start_video_thread(self):
        self.video_thread = HandGestureThread()
        self.video_thread.change_pixmap_signal.connect(self.update_webcam)
        self.video_thread.gesture_signal.connect(self.process_gesture)
        self.video_thread.start()

    def update_webcam(self, img):
        self.webcam_label.setPixmap(QPixmap.fromImage(img))

    def start_recording(self, target):
        self.recording_target = target
        self.status_label.setText("2초 후 저장됩니다... 포즈 유지!")
        self.status_label.setStyleSheet("color: orange; font-weight: bold;")
        QTimer.singleShot(2000, self.save_gesture)

    def save_gesture(self):
        if not self.current_fingers or self.current_hand_label == "None":
            self.status_label.setText("손을 찾을 수 없습니다.")
            self.status_label.setStyleSheet("color: red;")
            self.recording_target = None
            return

        data = (list(self.current_fingers), self.current_hand_label)
        txt = f"{data[1]} Hand {data[0]}"

        if self.recording_target == 'next':
            self.saved_data_next = data
            self.lbl_next.setText(txt)
            self.lbl_next.setStyleSheet("color: green; font-weight: bold;")
            self.status_label.setText("다음 페이지 제스처 저장 완료")
        else:
            self.saved_data_prev = data
            self.lbl_prev.setText(txt)
            self.lbl_prev.setStyleSheet("color: green; font-weight: bold;")
            self.status_label.setText("이전 페이지 제스처 저장 완료")

        self.status_label.setStyleSheet("color: black;")
        self.recording_target = None

    def process_gesture(self, fingers, label):
        self.current_fingers = fingers
        self.current_hand_label = label

        if self.recording_target or not self.can_trigger:
            return

        if not fingers or label == "None":
            return

        # Check Next
        if self.saved_data_next[0]:
            if fingers == self.saved_data_next[0] and label == self.saved_data_next[1]:
                self.req_next_page.emit()  # 메인으로 신호 발사
                self.activate_cooldown("다음 페이지 이동 ->")
                return

        # Check Prev
        if self.saved_data_prev[0]:
            if fingers == self.saved_data_prev[0] and label == self.saved_data_prev[1]:
                self.req_prev_page.emit()  # 메인으로 신호 발사
                self.activate_cooldown("<- 이전 페이지 이동")
                return

    def activate_cooldown(self, msg):
        self.can_trigger = False
        self.status_label.setText(msg)
        self.status_label.setStyleSheet("color: blue; font-weight: bold;")
        self.cooldown_timer.start()

    def reset_cooldown(self):
        self.can_trigger = True
        self.status_label.setText("대기 중...")
        self.status_label.setStyleSheet("color: black;")

    def closeEvent(self, event):
        if self.video_thread.isRunning():
            self.video_thread.stop()
        event.accept()

# 메인 슬라이드 뷰어 (UI)
class SlideWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Presentation Screen (Clean View)")
        self.resize(1024, 768)

        # 컨트롤러 창 생성 및 연결
        self.controller = ControllerWindow()
        self.controller.req_next_page.connect(self.next_page)
        self.controller.req_prev_page.connect(self.prev_page)
        self.controller.show()

        # PDF 속성
        self.doc = None
        self.page_count = 0
        self.current_page = 0
        self.page_pixmaps = []
        self.is_fullscreen = False
        self.normal_geometry = None

        # UI 구성
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: black;")
        self.image_label.setMinimumSize(1, 1)
        self.image_label.setSizePolicy(13, 13)

        container = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.image_label)
        container.setLayout(layout)
        self.setCentralWidget(container)

        self._create_toolbar()
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ctrl+O를 눌러 PDF를 열거나 F11으로 전체화면모드와 창모드를 전환 할 수 있습니다.")

    def _create_toolbar(self):
        self.toolbar = QToolBar("Main Toolbar")
        self.toolbar.setMovable(False)
        self.addToolBar(self.toolbar)

        open_action = QAction("PDF 열기(Ctrl+O)", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.open_pdf)
        self.toolbar.addAction(open_action)

        # 툴바 버튼용 (단축키는 아래 keyPressEvent에서 별도 처리)
        full_action = QAction("전체화면(F11)", self)
        full_action.triggered.connect(self.toggle_fullscreen)
        self.toolbar.addAction(full_action)

    #  키보드 입력 강제 처리 함수
    def keyPressEvent(self, event):
        # F11 키를 누르면 토글
        if event.key() == Qt.Key_F11:
            self.toggle_fullscreen()
        # ESC 키를 눌러도 전체화면 해제
        elif event.key() == Qt.Key_Escape and self.is_fullscreen:
            self.toggle_fullscreen()
        else:
            super().keyPressEvent(event)

    def toggle_fullscreen(self):
        if not self.is_fullscreen:
            self.normal_geometry = self.saveGeometry()
            self.toolbar.hide()
            self.status_bar.hide()
            self.showFullScreen()
            self.is_fullscreen = True
        else:
            self.showNormal()
            if self.normal_geometry is not None:
                self.restoreGeometry(self.normal_geometry)
            self.toolbar.show()
            self.status_bar.show()
            self.is_fullscreen = False
        if self.doc: self.show_current_page()

    def open_pdf(self):
        path, _ = QFileDialog.getOpenFileName(self, "PDF 선택", "", "PDF (*.pdf)")
        if not path: return
        self.doc = fitz.open(path)
        self.page_count = self.doc.page_count
        self.current_page = 0
        self.page_pixmaps = [None] * self.page_count
        self.show_current_page()

    def _render_page(self, idx):
        if self.page_pixmaps[idx]: return self.page_pixmaps[idx]
        page = self.doc.load_page(idx)
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
        img = QImage(pix.samples, pix.width, pix.height, pix.stride, QImage.Format_RGB888)
        self.page_pixmaps[idx] = QPixmap.fromImage(img)
        return self.page_pixmaps[idx]

    def show_current_page(self):
        if not self.doc: return
        pix = self._render_page(self.current_page)
        sz = self.image_label.size()
        if sz.width() > 0:
            self.image_label.setPixmap(pix.scaled(sz, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.status_bar.showMessage(f"Page {self.current_page + 1}/{self.page_count}")

    def next_page(self):
        if self.doc and self.current_page < self.page_count - 1:
            self.current_page += 1
            self.show_current_page()

    def prev_page(self):
        if self.doc and self.current_page > 0:
            self.current_page -= 1
            self.show_current_page()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.doc: self.show_current_page()

    def closeEvent(self, event):
        self.controller.close()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = SlideWindow()
    viewer.show()
    sys.exit(app.exec_())