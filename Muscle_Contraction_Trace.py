import cv2
import numpy as np
import matplotlib.pyplot as plt

def analyse_contractions(video_path, frame_skip=5, resize_factor=0.5, applied_voltage=True, cooldown_period=True):
    """
    Best suited to videos recorded at 60 FPS.
    For videos recorded at 30 FPS, use frame_skip=2.
    For videos recorded at 60 FPS, use frame_skip=5.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file.")
        return [], [], 0

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * resize_factor)
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * resize_factor)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_length = total_frames / fps

    contraction_times = []
    ret, prev_frame = cap.read()
    if not ret:
        print("Error reading the first frame of the video.")
        cap.release()
        return [], [], 0

    prev_gray = cv2.cvtColor(cv2.resize(prev_frame, (frame_width, frame_height)), cv2.COLOR_BGR2GRAY)

    frame_idx = 0
    last_contraction_time = -cooldown_period

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_skip == 0:
            frame_gray = cv2.cvtColor(cv2.resize(frame, (frame_width, frame_height)), cv2.COLOR_BGR2GRAY)
            frame_diff = cv2.absdiff(frame_gray, prev_gray)
            _, thresh = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)
            motion_intensity = np.sum(thresh)
            motion_threshold = 500

            current_time = frame_idx / fps
            if motion_intensity > motion_threshold and (current_time - last_contraction_time) >= cooldown_period:
                contraction_times.append(current_time)
                last_contraction_time = current_time

            prev_gray = frame_gray

        frame_idx += 1

    cap.release()

    voltage_trace = []
    times_with_zero = []
    for time in contraction_times:
        times_with_zero.extend([time, time])
        voltage_trace.extend([0, applied_voltage])

    return times_with_zero, voltage_trace, video_length

# User-defined parameters
applied_voltage = 7 # Enter the input voltage and ensure that the cooldown_period, in seconds, is lower than the expected time-period.
cooldown_period = 0.5

video_path = 'IMG_3295.MOV' # This is an example video path. The user is free to input their own videos in any format.
contraction_times_opt, voltage_trace_opt, video_length = analyse_contractions(video_path, applied_voltage=applied_voltage, cooldown_period=cooldown_period)

# Calculate the number of contractions and the rate of contraction (Hz)
number_of_contractions = len(contraction_times_opt) // 2
rate_of_contraction = number_of_contractions / video_length

plt.figure(figsize=(10, 6))
plt.stem(contraction_times_opt, voltage_trace_opt, linefmt='g-', basefmt='g-', markerfmt='')
plt.xlabel('Time (s)')
plt.ylabel('Applied Voltage (V)')
plt.title('Rate of Contractions with Applied Voltage')
plt.ylim(0, applied_voltage + 1)
plt.grid(True)

plt.text(0.05, 0.97, f'Number of Contractions: {number_of_contractions}\nRate of Contraction: {rate_of_contraction:.2f} Hz',
         transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')

plt.show()
