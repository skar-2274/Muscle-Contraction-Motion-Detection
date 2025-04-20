import cv2
import numpy as np
import matplotlib.pyplot as plt

roi_box = None  # Global to store the selected ROI

def draw_roi(event, x, y, flags, param):
    global roi_box, ref_img, temp_img, selecting, start_pt
    if event == cv2.EVENT_LBUTTONDOWN:
        selecting = True
        start_pt = (x, y)
    elif event == cv2.EVENT_MOUSEMOVE and selecting:
        temp_img = ref_img.copy()
        cv2.rectangle(temp_img, start_pt, (x, y), (0, 255, 0), 2)
        cv2.imshow("Select ROI", temp_img)
    elif event == cv2.EVENT_LBUTTONUP:
        selecting = False
        end_pt = (x, y)
        roi_box = (min(start_pt[0], end_pt[0]), min(start_pt[1], end_pt[1]),
                   abs(start_pt[0] - end_pt[0]), abs(start_pt[1] - end_pt[1]))
        cv2.rectangle(ref_img, start_pt, end_pt, (0, 255, 0), 2)
        cv2.imshow("Select ROI", ref_img)

def select_reference_frame(video_path, skip_frames=0):
    global ref_img, temp_img, selecting, start_pt, roi_box
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_num = skip_frames
    selected_frame = None
    paused = True

    print("Instructions:")
    print("  ENTER: Select current frame as reference")
    print("  P: Play/Pause")
    print("  A/D: Step backward / forward")
    print("  ESC: Exit\n")

    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            print("Reached end of video.")
            break

        frame_display = frame.copy()
        cv2.putText(frame_display, f"Frame: {frame_num}/{total_frames - 1}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Select Reference Frame', frame_display)

        key = cv2.waitKey(0 if paused else 30) & 0xFF

        if key == 13:  # ENTER
            selected_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            print(f"Selected frame {frame_num} as reference.")

            # Let user select ROI
            ref_img = frame.copy()
            temp_img = ref_img.copy()
            selecting = False
            start_pt = None

            cv2.namedWindow("Select ROI")
            cv2.setMouseCallback("Select ROI", draw_roi)
            print("Draw ROI with mouse. Press ENTER when done, or ESC to cancel.")
            while True:
                cv2.imshow("Select ROI", ref_img)
                roi_key = cv2.waitKey(1) & 0xFF
                if roi_key == 13:  # ENTER
                    print(f"ROI selected: {roi_box}")
                    break
                elif roi_key == 27:  # ESC
                    print("ROI selection cancelled.")
                    roi_box = None
                    break
            cv2.destroyWindow("Select ROI")
            break

        elif key == 27:
            print("Cancelled frame selection.")
            break
        elif key == ord('p'):
            paused = not paused
        elif key == ord('a'):
            frame_num = max(skip_frames, frame_num - 1)
        elif key == ord('d'):
            frame_num = min(total_frames - 1, frame_num + 1)
        elif not paused:
            frame_num += 1
            if frame_num >= total_frames:
                print("Reached end of video.")
                break

    cap.release()
    cv2.destroyAllWindows()
    return selected_frame

def extract_grayscale_frames(video_path, skip_frames=0):
    cap = cv2.VideoCapture(video_path)
    frames = []
    current_frame = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if current_frame >= skip_frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if roi_box:
                x, y, w, h = roi_box
                gray = gray[y:y+h, x:x+w]
            frames.append(gray)
        current_frame += 1

    cap.release()
    return frames

def motion_analysis(frames, reference_frame):
    amplitudes = []
    ref = reference_frame.astype(np.float32)
    if roi_box:
        x, y, w, h = roi_box
        ref = ref[y:y+h, x:x+w]

    for frame in frames:
        current = frame.astype(np.float32)
        diff = cv2.absdiff(current, ref)
        amplitudes.append(np.mean(diff))

    return np.array(amplitudes)

def normalise_signal(signal):
    return (signal - np.min(signal)) / (np.max(signal) - np.min(signal)) * 100

def main():
    video_path = "video_file.mp4"  # Change to your video path with any format
    skip_frames = 100 # Adjust in case of camera refocus

    ref_frame = select_reference_frame(video_path, skip_frames=skip_frames)
    if ref_frame is None:
        print("No reference frame selected. Exiting.")
        return

    frames = extract_grayscale_frames(video_path, skip_frames=skip_frames)
    motion = motion_analysis(frames, ref_frame)
    norm_motion = normalise_signal(motion)

    time_axis = np.arange(len(norm_motion)) / 30  # Adjust FPS if needed

    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 20,
        "axes.labelsize": 20,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20
    })

    plt.figure(figsize=(8, 5), dpi=100)
    plt.plot(time_axis, norm_motion, color='orange')
    plt.xlabel("Time (s)")
    plt.ylabel("Normalised Contraction\n(% change in area)")

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig("Contractions.png")
    plt.show()

if __name__ == "__main__":
    main()
