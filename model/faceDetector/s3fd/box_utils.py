import numpy as np
from itertools import product as product
import torch


# from torch.autograd import Function # Not used in this snippet

def nms_(dets, thresh):
    # ... (your nms_ code, ensure np.int becomes np.int32 if that was an issue) ...
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-5)  # Added epsilon for stability
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
    return np.array(keep).astype(np.int32)  # Explicitly np.int32


def decode(loc, priors, variances):
    loc_xy = loc[:, :2]
    loc_wh = loc[:, 2:]
    priors_xy = priors[:, :2]
    priors_wh = priors[:, 2:]

    # центр x, y
    center_xy_arg = loc_xy * variances[0] * priors_wh
    # print(f"DEBUG_DECODE: center_xy_arg (loc_xy * var[0] * priors_wh). Sample: {center_xy_arg.flatten()[:5].cpu().detach().numpy()}")
    decoded_center_xy = priors_xy + center_xy_arg

    # ширина, высота
    exp_arg = loc_wh * variances[1]
    # print(f"DEBUG_DECODE: exp_arg (loc_wh * var[1]). Sample: {exp_arg.flatten()[:5].cpu().detach().numpy()}")
    exp_term = torch.exp(exp_arg)
    # print(f"DEBUG_DECODE: exp_term (torch.exp(exp_arg)). Sample: {exp_term.flatten()[:5].cpu().detach().numpy()}")
    decoded_wh = priors_wh * exp_term

    boxes_center_form = torch.cat((decoded_center_xy, decoded_wh), 1)
    # print(f"DEBUG_DECODE: Boxes (center_x,y, w,h) before corner conversion. Sample: {boxes_center_form.flatten()[:20].cpu().detach().numpy()}")

    # Convert [center_x, center_y, w, h] to [x1, y1, x2, y2]
    # x1 = cx - w/2,  y1 = cy - h/2
    # x2 = x1 + w,    y2 = y1 + h
    final_boxes = torch.zeros_like(boxes_center_form)
    final_boxes[:, :2] = boxes_center_form[:, :2] - boxes_center_form[:, 2:] / 2  # x1, y1
    final_boxes[:, 2:] = final_boxes[:, :2] + boxes_center_form[:, 2:]  # x2, y2

    return final_boxes


def nms(boxes, scores, overlap=0.5, top_k=200):
    # ... (your nms code with the fix for torch.index_select) ...
    # This PyTorch NMS is used by Detect. No changes needed here for now unless it errors.
    keep = scores.new(scores.size(0)).zero_().long()
    if boxes.numel() == 0:
        return keep, 0
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = torch.mul(x2 - x1, y2 - y1)
    v, idx = scores.sort(0)
    idx = idx[-top_k:]
    w = boxes.new_empty(idx.size(0))  # Ensure w,h are sized for worst case if idx is not empty
    h = boxes.new_empty(idx.size(0))  # new_empty for uninitialized is fine for intermediate buffer
    count = 0
    while idx.numel() > 0:
        i = idx[-1]
        keep[count] = i
        count += 1
        if idx.size(0) == 1:
            break
        idx = idx[:-1]
        if idx.numel() == 0:  # check if idx became empty
            break

        # Ensure xx1,yy1,xx2,yy2 are created with correct size based on current idx
        xx1 = torch.index_select(x1, 0, idx)
        yy1 = torch.index_select(y1, 0, idx)
        xx2 = torch.index_select(x2, 0, idx)
        yy2 = torch.index_select(y2, 0, idx)

        xx1 = torch.clamp(xx1, min=x1[i].item())  # use .item() if x1[i] is 0-dim tensor
        yy1 = torch.clamp(yy1, min=y1[i].item())
        xx2 = torch.clamp(xx2, max=x2[i].item())
        yy2 = torch.clamp(yy2, max=y2[i].item())

        # Resize w and h based on the current size of xx2 (which is same as idx)
        # Using .resize_as_ might be tricky if w,h were created with .new() (empty).
        # Let's ensure w,h are correctly sized.
        # However, direct assignment is safer and more common now.
        current_w = xx2 - xx1
        current_h = yy2 - yy1

        current_w = torch.clamp(current_w, min=0.0)
        current_h = torch.clamp(current_h, min=0.0)
        inter = current_w * current_h
        rem_areas = torch.index_select(area, 0, idx)
        union = (rem_areas - inter) + area[i]
        IoU = inter / (union + 1e-5)  # Added epsilon for stability
        idx = idx[IoU.le(overlap)]
    return keep, count


class Detect(object):
    def __init__(self, num_classes=2,
                 top_k=750, nms_thresh=0.3, conf_thresh=0.05,
                 variance=[0.1, 0.2], nms_top_k=5000):
        self.num_classes = num_classes
        self.top_k = top_k
        self.nms_thresh = nms_thresh
        self.conf_thresh = conf_thresh
        self.variance = variance
        self.nms_top_k = nms_top_k

    def forward(self, loc_data, conf_data, prior_data):
        target_device = loc_data.device
        num = loc_data.size(0)
        num_priors = prior_data.size(0)

        conf_preds_transposed = conf_data.view(num, num_priors, self.num_classes).transpose(2, 1)

        batch_priors = prior_data.view(-1, num_priors, 4).expand(num, num_priors, 4)
        batch_priors = batch_priors.contiguous().view(-1, 4)

        decoded_boxes_tensor = decode(loc_data.view(-1, 4), batch_priors, self.variance)
        decoded_boxes_tensor = decoded_boxes_tensor.view(num, num_priors, 4)
        output = torch.zeros(num, self.num_classes, self.top_k, 5, device=target_device)

        boxes_current_batch = decoded_boxes_tensor[0].clone()
        conf_scores_current_batch = conf_preds_transposed[0].clone()

        for cl_idx in range(1, self.num_classes):  # Class loop (usually just class 1 for 'face')
            c_mask = conf_scores_current_batch[cl_idx].gt(self.conf_thresh)
            scores_filtered = conf_scores_current_batch[cl_idx][c_mask]

            if scores_filtered.numel() == 0:
                continue

            l_mask = c_mask.unsqueeze(1).expand(-1, 4).reshape(-1)
            boxes_filtered = boxes_current_batch.flatten()[l_mask].view(-1, 4)

            ids_kept, count_kept = nms(boxes_filtered, scores_filtered, self.nms_thresh, self.nms_top_k)

            actual_output_count = count_kept if count_kept < self.top_k else self.top_k
            if actual_output_count == 0:
                continue

            # Ensure ids_kept are used to index original filtered scores and boxes
            gathered_scores = scores_filtered[ids_kept[:actual_output_count]].unsqueeze(1)
            gathered_boxes = boxes_filtered[ids_kept[:actual_output_count]]

            if torch.backends.mps.is_available():
                torch.mps.synchronize()

            output[0, cl_idx, :actual_output_count] = torch.cat((gathered_scores, gathered_boxes), 1)

        if output.numel() > 0 and output.shape[2] > 0 and output.shape[1] > 1:  # Check if any actual detection filled
            if torch.backends.mps.is_available():
                torch.mps.synchronize()
            print(f"  {output[0, 1, 0, :].cpu().detach().numpy()}")  # Print first detection of class 1
        return output


class PriorBox(object):

    def __init__(self, input_size, feature_maps,
                 variance=[0.1, 0.2],
                 min_sizes=[16, 32, 64, 128, 256, 512],
                 steps=[4, 8, 16, 32, 64, 128],
                 clip=False):
        super(PriorBox, self).__init__()
        self.imh = input_size[0]  # e.g. scaled height
        self.imw = input_size[1]  # e.g. scaled width
        self.feature_maps = feature_maps
        self.variance = variance
        self.min_sizes = min_sizes
        self.steps = steps
        self.clip = clip

    def forward(self):
        mean = []
        for k, fmap in enumerate(self.feature_maps):
            feath = fmap[0]
            featw = fmap[1]
            for i_prod, j_prod in product(range(feath), range(featw)):  # Renamed i,j to avoid conflict
                f_kw = self.imw / self.steps[k]
                f_kh = self.imh / self.steps[k]
                cx = (j_prod + 0.5) / f_kw
                cy = (i_prod + 0.5) / f_kh
                s_kw = self.min_sizes[k] / self.imw
                s_kh = self.min_sizes[k] / self.imh
                mean += [cx, cy, s_kw, s_kh]

        # This creates a CPU tensor by default
        output_priors = torch.FloatTensor(mean).view(-1, 4)
        if self.clip:
            output_priors.clamp_(max=1, min=0)

        return output_priors
