import streamlit as st
import cv2
import numpy as np
import time
import pandas as pd
from scipy.signal import find_peaks
from scipy.signal import butter, sosfilt

# Load the Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to draw a rectangle on the frame
def draw_rectangle(frame, start_point, end_point):
    return cv2.rectangle(frame, start_point, end_point, (255, 0, 0), 2)

# Function to calculate heart rate from the PPG signal
def calculate_heart_rate(ppg_signal, fps):
    peaks, _ = find_peaks(ppg_signal[:-300], height=1)

    if len(peaks) > 1:
        peak_intervals = np.diff(peaks)
        avg_interval = np.mean(peak_intervals)
        heart_rate = 1 / ((avg_interval / fps) / 60)  # Assuming 30 frames per second
        return int(heart_rate), avg_interval / fps, peaks
    else:
        return 0, 0, []  # Not enough peaks to calculate heart rate

def main():
    st.title("Live PPG Heart Rate Monitor 🫀")

    # Start video capture
    video_capture = cv2.VideoCapture(0)

    # Create placeholders for video and heart rate
    frame_placeholder = st.empty()
    heart_rate_placeholder = st.empty()

    # Parameters
    R_values = []
    G_values = []
    all_R_values = []
    all_G_values = []
    all_ppg_signals = []
    fps = video_capture.get(cv2.CAP_PROP_FPS)

    # Streamlit layout with columns
    col1, col2 = st.columns([10, 10])  # Adjust column widths as needed

    with col1:
        st.header("Video Feed")
        frame_placeholder = st.empty()
        st.header("Heart Rate Information")
        heart_rate_placeholder = st.empty()
        
    with col2:
        st.header("Signal Plots")
        rgb_graph_placeholder = st.empty()
        ppg_graph_placeholder = st.empty()
        
    while True:
        ret, frame = video_capture.read()
        if not ret:
            st.write("Failed to capture video")
            break

        # Convert the frame to RGB (Streamlit uses RGB format)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert to grayscale for face detection
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

        # Initialize R and G to default values
        R = 0
        G = 0

        # If faces are detected, extract the first detected face
        for (x, y, w, h) in faces:
            frame_rgb = draw_rectangle(frame_rgb, (x, y), (x+w, y+h))
            face_roi = frame[y:y+h, x:x+w]
            mean_rgb = cv2.mean(face_roi)
            R = mean_rgb[2]
            G = mean_rgb[1]
            break

        # Collect the R and G values
        R_values.append(R)
        G_values.append(G)
        all_R_values.append(R)
        all_G_values.append(G)

        # Display the frame in Streamlit
        frame_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)

        # Calculate the PPG signal after collecting enough data
        if len(R_values) > 1:
            ppg_signal = np.array(G_values) / (np.array(R_values))
            all_ppg_signals.extend(ppg_signal.tolist())

            heart_rate, avg_interval, peaks = calculate_heart_rate(all_ppg_signals, fps)

            # Update the heart rate display
            heart_rate_placeholder.write(f"Estimated Heart Rate: {heart_rate} BPM\n AVG peak interval: {avg_interval:.2f} s\n FPS: {fps:.2f}")

            # Plot RGB values using Streamlit
            rgb_data_to_plot = pd.DataFrame({
                'R': all_R_values,
                'G': all_G_values
            })
            rgb_graph_placeholder.line_chart(rgb_data_to_plot)

            # Plot PPG signal using Streamlit
            ppg_data_to_plot = pd.DataFrame({'PPG Signal': ppg_signal})
            ppg_graph_placeholder.line_chart(ppg_data_to_plot)


    video_capture.release()

if __name__ == "__main__":
    main()
