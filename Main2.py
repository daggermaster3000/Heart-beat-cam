import streamlit as st
import cv2
import numpy as np
from scipy.signal import butter, filtfilt

# Set up the Streamlit app
st.title("GRGB rPPG Heart Rate Estimation with Face Detection 🫀")
st.write("An app for real-time heart rate estimation using the GRGB rPPG method with live PPG graph and face detection")

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Define video input (webcam)
cap = cv2.VideoCapture(0)

# Parameters for Butterworth bandpass filter
fs = 30.0  # Sample rate of the video in Hz
lowcut = 0.75  # Low cutoff frequency in Hz
highcut = 4.0  # High cutoff frequency in Hz

def butter_bandpass(lowcut, highcut, fs, order=6):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype="band")
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=6):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

# Function to calculate rPPG signal based on GRGB method
def calculate_grgb_rppg(roi):
    roi_mean = np.mean(roi, axis=(0, 1))
    g, r, b = roi_mean[1], roi_mean[2], roi_mean[0]
    gr_ratio = g / (r + 1e-6)  # Avoid division by zero
    gb_ratio = g / (b + 1e-6)  # Avoid division by zero
    return gr_ratio + gb_ratio, g, r, b

# Start video stream
if cap.isOpened():
    frames = []
    timestamps = []
    pixels = {"G": [], "R": [], "B": []}
    
    # Define placeholders for the video feed, output, and graphs
    frame_placeholder = st.empty()
    heart_rate_placeholder = st.empty()

    # Loading spinner while gathering initial samples
    with st.spinner("Gathering initial samples... Please wait."):
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert frame to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces in the frame
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
            
            # If a face is detected, use the face as the ROI
            if len(faces) > 0:
                x, y, w, h = faces[0]  # Use the first detected face
                roi = frame[y:y+h, x:x+w]
                
                # Draw a rectangle around the detected face (for visualization)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Calculate the rPPG signal
                rppg_signal, g, r, b = calculate_grgb_rppg(roi)
                
                frames.append(rppg_signal)
                timestamps.append(len(timestamps) / fs)  # Estimate timestamps for each frame
                
                # Append RGB values to pixels dictionary
                pixels["G"].append(g)
                pixels["R"].append(r)
                pixels["B"].append(b)
                
                # Exit the loading spinner once enough frames are collected
                if len(frames) > int(fs * 10):  # Approx 10 seconds worth of frames
                    break

    # Signal plots section
    st.header("rPPG Signal Plot")
    ppg_chart = st.line_chart()
    st.header("Pixel signal Plot")
    pixel_chart = st.line_chart()

    # Now proceed with heart rate estimation and live PPG graphing
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
        
        if len(faces) > 0:
            x, y, w, h = faces[0]
            roi = frame[y:y+h, x:x+w]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            rppg_signal, g, r, b = calculate_grgb_rppg(roi)
            frames.append(rppg_signal)
            timestamps.append(len(timestamps) / fs)
            
            # Append RGB values to pixels dictionary
            pixels["G"].append(g)
            pixels["R"].append(r)
            pixels["B"].append(b)
            
            # Normalize the rPPG signal
            normalized_signal = (frames - np.mean(frames)) / np.std(frames)
            
            # Filter the signal
            filtered_signal = bandpass_filter(normalized_signal, lowcut, highcut, fs)
            
            # Estimate heart rate
            heart_rate_frequency = np.fft.rfftfreq(len(filtered_signal), d=1.0/fs)
            heart_rate_spectrum = np.abs(np.fft.rfft(filtered_signal))
            heart_rate_peak = heart_rate_frequency[np.argmax(heart_rate_spectrum)]
            estimated_heart_rate = heart_rate_peak * 60.0  # Convert to bpm
            
            # Display heart rate and video feed
            heart_rate_placeholder.write(f"Estimated Heart Rate: {estimated_heart_rate:.2f} bpm")
            frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            # Update live PPG chart
            ppg_chart.line_chart(filtered_signal[-int(fs * 5):])  # Display the last 5 seconds of data
            
            # Update live RGB channels chart with the last 5 seconds of data
            pixel_chart.line_chart({
                "Green": pixels["G"][-int(fs * 5):],
                "Red": pixels["R"][-int(fs * 5):],
                "Blue": pixels["B"][-int(fs * 5):]
            })
