import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from .box_utils import Detect, PriorBox  # Detect class is defined in box_utils


class L2Norm(nn.Module):
    def __init__(self, n_channels, scale):
        super(L2Norm, self).__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        init.constant_(self.weight, self.gamma)

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps
        x = torch.div(x, norm)
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out


class S3FDNet(nn.Module):
    def __init__(self, device='mps'):  # Changed default to 'mps' for consistency with user
        super(S3FDNet, self).__init__()
        self.device = torch.device(device)  # Use torch.device object

        # ... (rest of your layer definitions: self.vgg, self.L2Norms, self.extras, self.loc, self.conf)
        self.vgg = nn.ModuleList([
            nn.Conv2d(3, 64, 3, 1, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2, ceil_mode=True),
            nn.Conv2d(256, 512, 3, 1, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(512, 512, 3, 1, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(512, 1024, 3, 1, padding=6, dilation=6), nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, 1, 1), nn.ReLU(inplace=True),
        ])
        self.L2Norm3_3 = L2Norm(256, 10)
        self.L2Norm4_3 = L2Norm(512, 8)
        self.L2Norm5_3 = L2Norm(512, 5)
        self.extras = nn.ModuleList([
            nn.Conv2d(1024, 256, 1, 1), nn.Conv2d(256, 512, 3, 2, padding=1),
            nn.Conv2d(512, 128, 1, 1), nn.Conv2d(128, 256, 3, 2, padding=1),
        ])
        self.loc = nn.ModuleList([
            nn.Conv2d(256, 4, 3, 1, padding=1), nn.Conv2d(512, 4, 3, 1, padding=1),
            nn.Conv2d(512, 4, 3, 1, padding=1), nn.Conv2d(1024, 4, 3, 1, padding=1),
            nn.Conv2d(512, 4, 3, 1, padding=1), nn.Conv2d(256, 4, 3, 1, padding=1),
        ])
        self.conf = nn.ModuleList([
            nn.Conv2d(256, 4, 3, 1, padding=1), nn.Conv2d(512, 2, 3, 1, padding=1),
            nn.Conv2d(512, 2, 3, 1, padding=1), nn.Conv2d(1024, 2, 3, 1, padding=1),
            nn.Conv2d(512, 2, 3, 1, padding=1), nn.Conv2d(256, 2, 3, 1, padding=1),
        ])
        # All submodules will be moved to self.device when self.to(self.device) is called in S3FD.__init__

        self.softmax = nn.Softmax(dim=-1)
        self.detect = Detect()  # Detect() does not take device argument
        # self.priorbox is initialized in forward pass

    def forward(self, x):
        # Ensure input x is on the same device as the network's parameters
        if x.device != self.device:
            x = x.to(self.device)

        size = x.size()[2:]
        sources = list()
        loc_preds = list()  # Renamed from loc to avoid confusion with outer scope loc
        conf_preds = list()  # Renamed from conf

        # Apply vgg layers
        for k in range(16):
            x = self.vgg[k](x)
        s = self.L2Norm3_3(x)
        sources.append(s)

        for k in range(16, 23):
            x = self.vgg[k](x)
        s = self.L2Norm4_3(x)
        sources.append(s)

        for k in range(23, 30):
            x = self.vgg[k](x)
        s = self.L2Norm5_3(x)
        sources.append(s)

        for k in range(30, len(self.vgg)):  # fc6, fc7
            x = self.vgg[k](x)
        sources.append(x)

        for k, v_layer in enumerate(self.extras):  # Renamed v to v_layer
            x = F.relu(v_layer(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)

        # Apply multibox head to source layers
        loc_x_head = self.loc[0](sources[0])  # Renamed loc_x to loc_x_head
        conf_x_head = self.conf[0](sources[0])  # Renamed conf_x to conf_x_head

        max_conf, _ = torch.max(conf_x_head[:, 0:3, :, :], dim=1, keepdim=True)
        conf_x_processed = torch.cat((max_conf, conf_x_head[:, 3:, :, :]), dim=1)  # Renamed

        loc_preds.append(loc_x_head.permute(0, 2, 3, 1).contiguous())
        conf_preds.append(conf_x_processed.permute(0, 2, 3, 1).contiguous())

        for i in range(1, len(sources)):
            src = sources[i]  # Renamed x to src
            conf_preds.append(self.conf[i](src).permute(0, 2, 3, 1).contiguous())
            loc_preds.append(self.loc[i](src).permute(0, 2, 3, 1).contiguous())

        features_maps = []
        for i in range(len(loc_preds)):
            feat = []
            feat += [loc_preds[i].size(1), loc_preds[i].size(2)]
            features_maps += [feat]

        loc_cat = torch.cat([o.view(o.size(0), -1) for o in loc_preds], 1)
        conf_cat = torch.cat([o.view(o.size(0), -1) for o in conf_preds], 1)

        # PriorBox is created here in each forward pass. It uses 'size' (from input x).
        # Its output priors will be on CPU by default.
        with torch.no_grad():
            self.priorbox = PriorBox(size, features_maps)  # size is from x, so refers to scaled image size
            priors_cpu = self.priorbox.forward()  # This creates CPU tensor

        # Priors are moved to self.device before being passed to self.detect.forward
        priors_on_device = priors_cpu.type(x.dtype).to(self.device)  # type(x.data) might be problematic, use x.dtype

        output_detections = self.detect.forward(
            loc_cat.view(loc_cat.size(0), -1, 4),  # loc_cat is on self.device
            self.softmax(conf_cat.view(conf_cat.size(0), -1, 2)),  # result of softmax on self.device
            priors_on_device  # priors_on_device is now on self.device
        )
        if output_detections.numel() > 0 and output_detections.shape[2] > 0:  # Check if there are any detections
            print(f"  {output_detections[0, :, 0, :].cpu().detach().numpy()}")

        return output_detections