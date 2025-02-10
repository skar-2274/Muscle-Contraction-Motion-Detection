import cv2
import numpy as np
import matplotlib.pyplot as plt

def analyse_contractions(video_path, frame_skip=5, resize_factor=0.5, applied_voltage=True, cooldown_period=True):
    # Initialize video capture and check if the video file is accessible.
    # For 30 FPS set frame_skip=2 and for 60 FPS set frame_skip=5. Code is optimised for 60 FPS with all video formats supported by OpenCV.
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open the video file.")
        return [], [], 0

    # Extract video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * resize_factor)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * resize_factor)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_length = total_frames / fps

    contraction_times = []
    ret, prev_frame = cap.read()
    if not ret:
        print("Error: Could not read the first frame.")
        cap.release()
        return [], [], 0

    prev_gray = cv2.cvtColor(cv2.resize(prev_frame, (width, height)), cv2.COLOR_BGR2GRAY)

    frame_idx = 0
    last_contraction_time = -cooldown_period

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_skip == 0:
            current_gray = cv2.cvtColor(cv2.resize(frame, (width, height)), cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(current_gray, prev_gray)
            _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
            motion_intensity = np.sum(thresh)
            motion_threshold = 500

            current_time = frame_idx / fps
            if motion_intensity > motion_threshold and (current_time - last_contraction_time) >= cooldown_period:
                contraction_times.append(current_time)
                last_contraction_time = current_time

            prev_gray = current_gray

        frame_idx += 1

    cap.release()

    # Prepare the voltage trace for plotting
    voltage_trace = []
    plot_times = []
    for time in contraction_times:
        plot_times.extend([time, time])
        voltage_trace.extend([0, applied_voltage])

    return plot_times, voltage_trace, video_length

# Parameters (customisable)
applied_voltage = 7 # Input the voltage used during the stimulation process.
cooldown_period = 0.5 # Make this value less than time period of the expected out frequency as calculated by 1/T.
video_path = 'IMG_3295.MOV'  # Replace with your video file

# Analyse contractions
contraction_times, voltage_trace, video_length = analyse_contractions(video_path, applied_voltage=applied_voltage, cooldown_period=cooldown_period)

# Calculate contractions stats
contractions_count = len(contraction_times) // 2
contraction_rate = contractions_count / video_length

plt.figure(figsize=(10, 6))
plt.stem(contraction_times, voltage_trace, linefmt='g-', basefmt='g-', markerfmt='')
plt.xlabel('Time (s)')
plt.ylabel('Applied Voltage (V)')
plt.title('Contraction Activity with Applied Voltage')
plt.ylim(0, applied_voltage + 1)
plt.grid(True)

plt.text(0.05, 0.97, f'Number of Contractions: {contractions_count}\nRate of Contractions: {contraction_rate:.2f} Hz',
         transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')

plt.show()
