import cv2
import numpy as np
from fer import FER  # Install dengan 'pip install fer'
import streamlit as st

def detect_motion_and_expression():
    st.title("Motion and Emotion Detection")

    # Inisialisasi webcam
    cap = cv2.VideoCapture(0)  # 0 untuk webcam default

    # Inisialisasi detektor ekspresi wajah
    emotion_detector = FER()

    # Baca frame pertama
    ret, frame = cap.read()
    if not ret:
        st.error("Gagal mengakses webcam.")
        return

    prev_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Stream video ke Streamlit
    frame_placeholder = st.empty()

    while True:
        # Baca frame baru
        ret, frame = cap.read()
        if not ret:
            st.error("Gagal membaca frame.")
            break

        # Konversi ke grayscale untuk deteksi gerakan
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Hitung perbedaan antara frame untuk deteksi gerakan
        frame_diff = cv2.absdiff(prev_frame, gray)
        
        # Terapkan threshold
        thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)[1]

        # Dilasi untuk mengisi lubang-lubang kecil
        thresh = cv2.dilate(thresh, None, iterations=2)

        # Temukan kontur pada threshold image
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        motion_detected = False
        for contour in contours:
            if cv2.contourArea(contour) > 500:  # Anda bisa menyesuaikan nilai ini
                (x, y, w, h) = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                motion_detected = True

        # Deteksi wajah dan ekspresi menggunakan model FER
        faces = emotion_detector.detect_emotions(frame)

        for face in faces:
            # Gambar kotak di sekitar wajah
            x, y, w, h = face["box"]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Dapatkan ekspresi dengan confidence tertinggi
            emotion, score = emotion_detector.top_emotion(frame[y:y+h, x:x+w])
            
            # Tampilkan emosi di atas wajah
            if score > 0.5:  # Threshold untuk confidence
                cv2.putText(frame, f'{emotion}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        # Update frame sebelumnya
        prev_frame = gray

        # Tampilkan frame pada Streamlit
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Ubah dari BGR ke RGB
        frame_placeholder.image(frame, channels="RGB")

        # Jika ingin berhenti, kita bisa gunakan tombol 'q' atau perintah lainnya (sesuaikan dengan Streamlit)

        # Berhenti jika tombol 'Stop' ditekan
        if st.button("Stop"):
            break

    # Bersihkan
    cap.release()

if __name__ == "__main__":
    st.sidebar.title("Control Panel")
    if st.sidebar.button("Mulai Deteksi"):
        detect_motion_and_expression()
