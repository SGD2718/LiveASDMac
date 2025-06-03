import time, os, sys, subprocess
import numpy as np
import cv2
import torch
# from torchvision import transforms # Not used in S3FD class
from .nets import S3FDNet
from .box_utils import nms_
import coremltools as ct

# Corrected PATH_WEIGHT definition for robustness
# It should be relative to this file if sfd_face.pth is in the same directory
# Or use the original if it relies on CWD for the gdown logic.
# For simplicity, let's assume sfd_face.pth is in the same directory as this __init__.py
# or gdown places it where PATH_WEIGHT (relative to CWD) can find it.
_S3FD_DIR = os.path.dirname(os.path.abspath(__file__))
PATH_WEIGHT_CANDIDATE = os.path.join(_S3FD_DIR, 'sfd_face.pth')

if os.path.isfile(PATH_WEIGHT_CANDIDATE):
    PATH_WEIGHT = PATH_WEIGHT_CANDIDATE
else:
    # Fallback to original logic if local .pth not found (might rely on CWD for gdown)
    PATH_WEIGHT = 'model/faceDetector/s3fd/sfd_face.pth'
    if not os.path.isfile(PATH_WEIGHT):
        Link = "1KafnHz7ccT-3IyddBsL5yi2xGtxAKypt"
        # Ensure PATH_WEIGHT is the target for gdown
        gdown_target_path = PATH_WEIGHT
        # If PATH_WEIGHT is 'model/faceDetector/s3fd/sfd_face.pth' and CWD is Light-ASD-main, this is fine.
        # If gdown needs to place it directly into _S3FD_DIR, adjust gdown_target_path.
        # Assuming gdown target path is what original script implies for simplicity:
        os.makedirs(os.path.dirname(gdown_target_path), exist_ok=True)  # Ensure directory exists
        cmd = "gdown --id %s -O %s" % (Link, gdown_target_path)
        subprocess.call(cmd, shell=True, stdout=None)
        if os.path.isfile(gdown_target_path):
            PATH_WEIGHT = gdown_target_path  # Update PATH_WEIGHT if downloaded
        else:
            print(f"FATAL: gdown failed to download to {gdown_target_path}")

img_mean = np.array([104., 117., 123.])[:, np.newaxis, np.newaxis].astype('float32')


class S3FD():
    used_shapes = set()

    def __init__(self, device='mps'):
        self.device = torch.device(device)  # Explicit torch.device object

        self.net = S3FDNet(device=self.device).to(self.device)

        try:
            state_dict = torch.load(PATH_WEIGHT, map_location=self.device)
            self.net.load_state_dict(state_dict)
            self.net.eval()
        except Exception as e:
            print(f"FATAL_S3FD_INIT: Error loading S3FD weights: {e}")
            raise
        # print('[S3FD] finished loading (%.4f sec)' % (time.time() - tstamp))

    def detect_faces(self, image, conf_th=0.8, scales=[1]):
        w, h = image.shape[1], image.shape[0]
        bboxes = np.empty(shape=(0, 5))

        with torch.no_grad():
            for s_idx, s_val in enumerate(scales):
                scaled_img = cv2.resize(image, dsize=(0, 0), fx=s_val, fy=s_val, interpolation=cv2.INTER_LINEAR)

                # Preprocessing
                proc_img = np.swapaxes(scaled_img, 1, 2)
                proc_img = np.swapaxes(proc_img, 1, 0)
                proc_img = proc_img[[2, 1, 0], :, :]  # Input 'image' is RGB, converted to BGR
                proc_img = proc_img.astype('float32')
                proc_img -= img_mean  # img_mean is BGR
                proc_img = proc_img[[2, 1, 0], :, :]  # Converted back to RGB for the model

                x = torch.from_numpy(proc_img.copy()).unsqueeze(0).to(self.device)

                y = self.net(x)  # Model inference

                detections = y.data  # detections tensor is on self.device

                scale_tensor = torch.tensor([w, h, w, h], device=self.device, dtype=torch.float32)

                num_potential_dets = 0
                for i in range(detections.size(1)):  # Class loop
                    j = 0  # Detection loop for this class
                    while j < detections.size(2) and detections[0, i, j, 0] > conf_th:
                        num_potential_dets += 1
                        score_tensor = detections[0, i, j, 0]
                        coords_norm_mps = detections[0, i, j, 1:]

                        pt_on_device = coords_norm_mps * scale_tensor
                        pt = pt_on_device.cpu().numpy()  # Transfer result to CPU for numpy
                        bbox = (pt[0], pt[1], pt[2], pt[3], score_tensor.item())
                        bboxes = np.vstack((bboxes, bbox))
                        j += 1

            if bboxes.shape[0] > 0:
                keep = nms_(bboxes, 0.1)
                bboxes = bboxes[keep]

        return bboxes