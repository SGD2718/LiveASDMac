import coremltools
import cv2
import numpy as np
import argparse
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'
import sys
import torch  # To check for MPS availability
import torchvision

# Ensure the model directory is in the Python path if the script is run from outside
# For example, if this script is in Light-ASD-main/
# and the model is in Light-ASD-main/model/
# from model.faceDetector.s3fd import S3FD
# If the script is placed in Light-ASD-main, this should work directly.
# If you place it elsewhere, you might need to adjust sys.path:
# script_dir = os.path.dirname(os.path.abspath(__file__))
# project_root = os.path.abspath(os.path.join(script_dir, '..')) # Adjust if script is deeper
# sys.path.insert(0, project_root)

try:
    from model.faceDetector.s3fd import S3FD
except ImportError:
    print("Error: Could not import S3FD model.")
    print("Please ensure this script is in the root of the Light-ASD-main directory,")
    print("or that the 'model' directory is in your PYTHONPATH.")
    sys.exit(1)


def test_face_detection(image_path, output_path, device_str, conf_threshold, image_scale):
    """
    Loads an image, detects faces using the S3FD model, draws bounding boxes,
    and saves the output image.
    """
    print(f"--- Face Detection Test ---")
    print(f"Input image: {image_path}")
    print(f"Output image: {output_path}")
    print(f"Device: {device_str}")
    print(f"Confidence threshold: {conf_threshold}")
    print(f"Image scale for detector: {image_scale}")

    # Check if MPS is available if requested
    if device_str == 'mps' and not torch.mps.is_available():  # Corrected from torch.backends.mps.is_available()
        print("Warning: MPS requested but not available. Falling back to CPU.")
        device_str = 'cpu'
    elif device_str == 'mps':
        print("INFO: MPS device selected and available.")


    # Load the S3FD face detection model
    print(f"\nLoading S3FD model on {device_str}...")
    try:
        face_detector = S3FD(device=device_str)
        print("S3FD model loaded successfully.")
    except Exception as e:
        print(f"Error loading S3FD model: {e}")
        print("Please ensure 'model/faceDetector/s3fd/sfd_face.pth' exists or can be downloaded,")
        print("and that PyTorch is correctly installed.")
        sys.exit(1)

    # Load the input image
    print(f"\nLoading image from {image_path}...")
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image from {image_path}")
        sys.exit(1)

    original_height, original_width = image.shape[:2]
    print(f"Original image dimensions: {original_width}x{original_height}")

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    print(f"\nDetecting faces...")
    try:
        bboxes = face_detector.detect_faces(image_rgb, conf_th=conf_threshold, scales=[image_scale])
    except Exception as e:
        print(f"Error during face detection: {e}")
        sys.exit(1)

    print(f"Detected {len(bboxes)} faces.")

    output_image = image.copy()
    if len(bboxes) > 0: # Check if bboxes is not None and has entries
        for i, bbox in enumerate(bboxes):
            if bbox is None or len(bbox) < 5:
                print(f"  Skipping invalid bbox format: {bbox}")
                continue
            x1, y1, x2, y2, score = bbox
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            print(f"  Face {i + 1}: BBox=({x1},{y1})-({x2},{y2}), Score={score:.4f}")

            cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            text = f"Score: {score:.2f}"
            text_y_pos = y1 - 10 if y1 - 10 > 10 else y1 + 20
            cv2.putText(output_image, text, (x1, text_y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        print("No faces detected to draw.")


    print(f"\nSaving output image with detections to {output_path}...")
    try:
        cv2.imwrite(output_path, output_image)
        print(f"Successfully saved output to {output_path}")
    except Exception as e:
        print(f"Error saving image: {e}")

    print("\n--- Test Complete ---")


if __name__ == "__main__":
    # ADD THIS AT THE VERY TOP IF TESTING MPS FALLBACK
    # import os
    # os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    # print("DEBUG: PYTORCH_ENABLE_MPS_FALLBACK is SET" if os.getenv('PYTORCH_ENABLE_MPS_FALLBACK') == '1' else "DEBUG: PYTORCH_ENABLE_MPS_FALLBACK is NOT SET")

    parser = argparse.ArgumentParser(description="Test Script for S3FD Face Detection Model")
    parser.add_argument('--image_path', type=str, required=True,
                        help='Path to the input image file.')
    parser.add_argument('--output_path', type=str, default='detected_faces_output.jpg',
                        help='Path to save the output image with detected faces.')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'mps'],
                        help='Device to run the model on (e.g., \"cpu\", \"mps\"). Default is \"cpu\".')
    parser.add_argument('--conf_th', type=float, default=0.8,
                          help='Confidence threshold for face detections (0.0 to 1.0). Default is 0.8.')
    parser.add_argument('--scale', type=float, default=0.5,
                        help='Factor by which to scale the image before detection (e.g., 1.0 for original size, 0.5 for half size). The S3FD model in Columbia_test.py uses 0.25. Default is 1.0.')

    args = parser.parse_args()

    if not os.path.isfile(args.image_path):
        print(f"Error: Input image path not found: {args.image_path}")
        sys.exit(1)

    test_face_detection(args.image_path, args.output_path, args.device, args.conf_th, args.scale)


