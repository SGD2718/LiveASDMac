#!/usr/bin/env python
# live_asd_m2_mac.py

import cv2
import numpy
import sounddevice as sd
import torch
import python_speech_features
import argparse
import time
import collections

# Assuming the repository files are in the PYTHONPATH or same directory
from model.faceDetector import s3fd
from model.faceDetector.s3fd import S3FD
from ASD import ASD  # ASD class from ASD.py

# --- Configuration ---
SAMPLE_RATE = 16000
CHANNELS = 1
AUDIO_CHUNK_DURATION_S = 1.0  # Process 1-second audio chunks
VIDEO_FPS_TARGET = 25  # Target FPS for video processing
FACE_CROP_SIZE = (224, 224)  # As used in the model
VISUAL_FEATURE_SIZE_FOR_MODEL = (56, 56)  # ROI for visual features, as per paper/Columbia_test.py
SEQUENCE_LENGTH_FRAMES = 25  # Number of video frames / audio MFCCs for a sequence (e.g., 1 second at 25 FPS)

# For M2 Mac, try to use MPS for PyTorch
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# DEVICE = torch.device("cpu") # Fallback to CPU if needed

class LiveSpeakerDetector:
    def __init__(self, model_path):
        self.face_detector = S3FD(device=DEVICE)
        print(f"Face detector loaded on {DEVICE}.")

        self.asd_model = ASD(lr=0.001, lrDecay=0.95)  # lr/lrDecay not used for inference
        self.asd_model.loadParameters(model_path)
        self.asd_model = self.asd_model.to(DEVICE)
        self.asd_model.eval()
        print(f"ASD model loaded from {model_path} on {DEVICE}.")

        # Buffers for audio and video features for a single tracked face
        # In a more complex scenario, you'd have one set of buffers per tracked face
        self.audio_mfcc_buffer = collections.deque(
            maxlen=int(SEQUENCE_LENGTH_FRAMES * 4))  # MFCCs are 4x video frames typically
        self.video_face_buffer = collections.deque(maxlen=SEQUENCE_LENGTH_FRAMES)
        self._frame_counter_fps = 0
        self.audio_stream = None
        self.audio_buffer = collections.deque(
            maxlen=int(SAMPLE_RATE * (AUDIO_CHUNK_DURATION_S + 0.5)))  # Buffer slightly more audio

    def _audio_callback(self, indata, frames, time, status):
        if status:
            print(status)
        self.audio_buffer.extend(indata[:, 0])

    def start_audio_capture(self):
        print(f"Starting audio stream with {SAMPLE_RATE} Hz, {AUDIO_CHUNK_DURATION_S}s chunks.")
        self.audio_stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            callback=self._audio_callback,
            blocksize=int(SAMPLE_RATE * 0.1)  # Smaller blocksize for lower latency callback
        )
        self.audio_stream.start()

    def stop_audio_capture(self):
        if self.audio_stream:
            self.audio_stream.stop()
            self.audio_stream.close()
            print("Audio stream stopped.")

        # Inside LiveSpeakerDetector.process_frame in live_asd_m2_mac.py

    def process_frame(self, frame):
        image_numpy = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        current_conf_th = 0.5  # Keep permissive for now
        current_scales = [0.75]
        # This calls S3FD's detect_faces
        bboxes = self.face_detector.detect_faces(image_numpy, conf_th=current_conf_th, scales=current_scales)

        # --- VERY IMPORTANT DEBUG PRINT ---
        print(f"\n--- Frame {self._frame_counter_fps} in live_asd_m2_mac.py ---")  # Use the initialized counter
        print(f"S3FD detect_faces called with conf_th={current_conf_th}, scales={current_scales}")
        print(f"Number of bboxes RECEIVED by live_asd_m2_mac.py: {len(bboxes)}")
        if len(bboxes) > 0:
            print(f"First bbox received by live_asd_m2_mac.py: {bboxes[0]}")
        # --- END VERY IMPORTANT DEBUG PRINT ---

        active_speaker_found = False
        active_speaker_bbox = None  # This isn't used much yet, but good to have

        if len(bboxes) > 0:
            # If we get here, S3FD returned at least one face
            print("[ASD Pipeline] Processing detected faces...")
            bboxes_sorted = sorted(bboxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
            best_bbox = bboxes_sorted[0]

            frame_h, frame_w = frame.shape[:2]
            half_w = frame_w / 2
            half_h = frame_h / 2
            x1 = max(0, int(best_bbox[0] + half_w))
            y1 = max(0, int(best_bbox[1] + half_h))
            x2 = min(frame_w - 1, int(best_bbox[2] + half_w))
            y2 = min(frame_h - 1, int(best_bbox[3] + half_h))
            face_score = best_bbox[4]

            print(f"[ASD Pipeline] Best bbox (clipped): x1={x1}, y1={y1}, x2={x2}, y2={y2}, score={face_score:.4f}")

            if x2 > x1 and y2 > y1:  # Valid dimensions for crop
                face_crop = frame[y1:y2, x1:x2]
                if face_crop.size == 0:
                    print("[ASD Pipeline] Face crop is empty! Skipping ASD for this face.")
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)  # Orange
                    cv2.putText(frame, f"Empty Crop s:{face_score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 165, 255), 2)
                    return frame, False, None

                face_gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
                face_to_buffer = cv2.resize(face_gray, (112, 112))
                self.video_face_buffer.append(face_to_buffer)

                # Audio buffering logic (ensure this part is also working)
                required_audio_samples_for_sequence = int(
                    SAMPLE_RATE * (SEQUENCE_LENGTH_FRAMES / VIDEO_FPS_TARGET))  # Audio for the whole sequence
                if len(self.audio_buffer) >= required_audio_samples_for_sequence:  # Check if enough total audio is buffered
                    current_audio_segment_for_mfcc = numpy.array(
                        list(self.audio_buffer)[-required_audio_samples_for_sequence:])
                    if len(current_audio_segment_for_mfcc) > 400:  # Min samples for python_speech_features.mfcc
                        mfccs = python_speech_features.mfcc(
                            current_audio_segment_for_mfcc, SAMPLE_RATE, winlen=0.025, winstep=0.010, numcep=13
                        )
                        self.audio_mfcc_buffer.clear()  # Use MFCCs for the current full sequence
                        self.audio_mfcc_buffer.extend(mfccs)
                        # print(f"[ASD Pipeline] Generated {len(mfccs)} MFCC vectors.") # Optional

                expected_audio_mfccs = int(SEQUENCE_LENGTH_FRAMES / VIDEO_FPS_TARGET * 100 * 0.9)
                print(f"[ASD Pipeline] Buffers: Video={len(self.video_face_buffer)}/{SEQUENCE_LENGTH_FRAMES}, " +
                      f"Audio MFCCs={len(self.audio_mfcc_buffer)} (target ~{expected_audio_mfccs:.0f})")

                if len(self.video_face_buffer) == SEQUENCE_LENGTH_FRAMES and \
                        len(self.audio_mfcc_buffer) >= expected_audio_mfccs:
                    print("[ASD Pipeline] Buffers full. Proceeding to ASD MODEL INFERENCE.")

                    # ... (The ASD model inference logic you pasted previously, which defines avg_speaking_score and active_speaker_found) ...
                    # Make sure this block is exactly as provided, calculating avg_speaking_score

                    video_tensor_list = [torch.from_numpy(np_array).float() for np_array in self.video_face_buffer]
                    video_input_tensor = torch.stack(video_tensor_list).unsqueeze(0).to(DEVICE)
                    audio_input_tensor = torch.FloatTensor(numpy.array(self.audio_mfcc_buffer)).unsqueeze(0).to(
                        DEVICE)

                    target_mfcc_len = video_input_tensor.shape[1] * 4
                    if audio_input_tensor.shape[1] < target_mfcc_len:
                        padding_len = target_mfcc_len - audio_input_tensor.shape[1]
                        padding = torch.zeros(
                            (audio_input_tensor.shape[0], padding_len, audio_input_tensor.shape[2]), device=DEVICE)
                        audio_input_tensor = torch.cat((audio_input_tensor, padding), dim=1)
                    elif audio_input_tensor.shape[1] > target_mfcc_len:
                        audio_input_tensor = audio_input_tensor[:, :target_mfcc_len, :]

                    print(
                        f"[ASD Pipeline] Visual tensor shape: {video_input_tensor.shape}, Audio tensor shape: {audio_input_tensor.shape}")

                    avg_speaking_score = 0.0  # Initialize
                    # active_speaker_found = False # Already initialized earlier

                    with torch.no_grad():
                        outsAV, _ = self.asd_model.model(audioFeature=audio_input_tensor,
                                                         visualFeature=video_input_tensor)
                        x_fc = self.asd_model.lossAV.FC(outsAV)
                        pred_scores_softmax = torch.softmax(x_fc, dim=-1)
                        speaking_scores = pred_scores_softmax[:, 1]
                        avg_speaking_score = torch.mean(speaking_scores).item()

                    threshold = 0.5
                    active_speaker_found = avg_speaking_score > threshold
                    print(
                        f"[ASD Pipeline] ASD Model Score: {avg_speaking_score:.4f}, Active: {active_speaker_found}")
                    # ... (Drawing logic for Active/Not Active) ...
                    if active_speaker_found:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"Active: {avg_speaking_score:.2f}", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    else:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(frame, f"Not Active: {avg_speaking_score:.2f}", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                else:  # Buffers not full for inference yet
                    print("[ASD Pipeline] Buffering face and audio data...")
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)  # Yellow
                    cv2.putText(frame, f"Buffering... s:{face_score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 255, 255), 2)
            else:  # Invalid bbox dimensions from S3FD (after clipping, x1>=x2 or y1>=y2)
                print(
                    f"[ASD Pipeline] Invalid bbox dimensions after clipping: x1={x1}, y1={y1}, x2={x2}, y2={y2}. No crop or ASD.")
        else:  # No bboxes returned by S3FD
            print("[ASD Pipeline] No faces detected by S3FD to process.")

        # Increment the frame counter (make sure self._frame_counter_fps is initialized in __init__)
        self._frame_counter_fps = (self._frame_counter_fps + 1) % 300  # Print every 300 frames or so

        return frame, active_speaker_found, active_speaker_bbox

def main(args):
    detector = LiveSpeakerDetector(model_path=args.model_path)

    cap = cv2.VideoCapture(args.camera_id)
    if not cap.isOpened():
        print(f"Error: Could not open video stream from camera {args.camera_id}.")
        return

    cap.set(cv2.CAP_PROP_FPS, VIDEO_FPS_TARGET)
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Attempted to set camera FPS to {VIDEO_FPS_TARGET}, actual FPS: {actual_fps}")

    # Update SEQUENCE_LENGTH_FRAMES if FPS is different and we want 1 sec
    global SEQUENCE_LENGTH_FRAMES
    if actual_fps > 0:
        SEQUENCE_LENGTH_FRAMES = int(actual_fps * AUDIO_CHUNK_DURATION_S)
        detector.audio_mfcc_buffer = collections.deque(maxlen=int(SEQUENCE_LENGTH_FRAMES * 4))
        detector.video_face_buffer = collections.deque(maxlen=SEQUENCE_LENGTH_FRAMES)
        print(
            f"Adjusted sequence length to {SEQUENCE_LENGTH_FRAMES} frames based on actual FPS and {AUDIO_CHUNK_DURATION_S}s duration.")

    detector.start_audio_capture()

    frame_count = 0
    start_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Can't receive frame (stream end?). Exiting ...")
                break

            processed_frame, found, box = detector.process_frame(frame)
            print("Found speaker: ", found)

            frame_count += 1
            if frame_count % 100 == 0:
                elapsed_time = time.time() - start_time
                current_processing_fps = frame_count / elapsed_time
                print(f"Processing FPS: {current_processing_fps:.2f}")

            cv2.imshow('Live Active Speaker Detection (M2 Mac)', processed_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        detector.stop_audio_capture()
        cap.release()
        cv2.destroyAllWindows()
        print("Released resources.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Live Active Speaker Detection on M2 Mac")
    parser.add_argument('--model_path', type=str, default="weight/pretrain_AVA_CVPR.pt",
                        help='Path to the pretrained ASD model weights (e.g., pretrain_AVA_CVPR.pt or finetuning_TalkSet.pt from the repo)')
    parser.add_argument('--camera_id', type=int, default=0, help='Camera ID for OpenCV VideoCapture')

    cli_args = parser.parse_args()
    main(cli_args)