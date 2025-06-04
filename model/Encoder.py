import torch
import torch.nn as nn

import os

#os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'


class Audio_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Audio_Block, self).__init__()

        self.relu = nn.ReLU()

        self.m_3 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 1), padding=(1, 0), bias=False)
        self.bn_m_3 = nn.BatchNorm2d(out_channels, momentum=0.01, eps=0.001)
        self.t_3 = nn.Conv2d(out_channels, out_channels, kernel_size=(1, 3), padding=(0, 1), bias=False)
        self.bn_t_3 = nn.BatchNorm2d(out_channels, momentum=0.01, eps=0.001)

        self.m_5 = nn.Conv2d(in_channels, out_channels, kernel_size=(5, 1), padding=(2, 0), bias=False)
        self.bn_m_5 = nn.BatchNorm2d(out_channels, momentum=0.01, eps=0.001)
        self.t_5 = nn.Conv2d(out_channels, out_channels, kernel_size=(1, 5), padding=(0, 2), bias=False)
        self.bn_t_5 = nn.BatchNorm2d(out_channels, momentum=0.01, eps=0.001)

        self.last = nn.Conv2d(out_channels, out_channels, kernel_size=(1, 1), padding=(0, 0), bias=False)
        self.bn_last = nn.BatchNorm2d(out_channels, momentum=0.01, eps=0.001)

    def forward(self, x):
        x_3 = self.relu(self.bn_m_3(self.m_3(x)))
        x_3 = self.relu(self.bn_t_3(self.t_3(x_3)))

        x_5 = self.relu(self.bn_m_5(self.m_5(x)))
        x_5 = self.relu(self.bn_t_5(self.t_5(x_5)))

        x = x_3 + x_5
        x = self.relu(self.bn_last(self.last(x)))

        return x


class Visual_Block(nn.Module):
    def __init__(self, in_channels, out_channels, is_down=False):
        super(Visual_Block, self).__init__()

        self.relu = nn.ReLU()

        if is_down:
            self.s_3 = nn.Conv3d(in_channels, out_channels, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1),
                                 bias=False)
            self.bn_s_3 = nn.BatchNorm3d(out_channels, momentum=0.01, eps=0.001)
            self.t_3 = nn.Conv3d(out_channels, out_channels, kernel_size=(3, 1, 1), padding=(1, 0, 0), bias=False)
            self.bn_t_3 = nn.BatchNorm3d(out_channels, momentum=0.01, eps=0.001)

            self.s_5 = nn.Conv3d(in_channels, out_channels, kernel_size=(1, 5, 5), stride=(1, 2, 2), padding=(0, 2, 2),
                                 bias=False)
            self.bn_s_5 = nn.BatchNorm3d(out_channels, momentum=0.01, eps=0.001)
            self.t_5 = nn.Conv3d(out_channels, out_channels, kernel_size=(5, 1, 1), padding=(2, 0, 0), bias=False)
            self.bn_t_5 = nn.BatchNorm3d(out_channels, momentum=0.01, eps=0.001)
        else:
            self.s_3 = nn.Conv3d(in_channels, out_channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False)
            self.bn_s_3 = nn.BatchNorm3d(out_channels, momentum=0.01, eps=0.001)
            self.t_3 = nn.Conv3d(out_channels, out_channels, kernel_size=(3, 1, 1), padding=(1, 0, 0), bias=False)
            self.bn_t_3 = nn.BatchNorm3d(out_channels, momentum=0.01, eps=0.001)

            self.s_5 = nn.Conv3d(in_channels, out_channels, kernel_size=(1, 5, 5), padding=(0, 2, 2), bias=False)
            self.bn_s_5 = nn.BatchNorm3d(out_channels, momentum=0.01, eps=0.001)
            self.t_5 = nn.Conv3d(out_channels, out_channels, kernel_size=(5, 1, 1), padding=(2, 0, 0), bias=False)
            self.bn_t_5 = nn.BatchNorm3d(out_channels, momentum=0.01, eps=0.001)

        self.last = nn.Conv3d(out_channels, out_channels, kernel_size=(1, 1, 1), padding=(0, 0, 0), bias=False)
        self.bn_last = nn.BatchNorm3d(out_channels, momentum=0.01, eps=0.001)

    def forward(self, x):

        x_3 = self.relu(self.bn_s_3(self.s_3(x)))
        x_3 = self.relu(self.bn_t_3(self.t_3(x_3)))

        x_5 = self.relu(self.bn_s_5(self.s_5(x)))
        x_5 = self.relu(self.bn_t_5(self.t_5(x_5)))

        x = x_3 + x_5

        x = self.relu(self.bn_last(self.last(x)))

        return x


# Assuming Visual_Block class is defined in the same file (Encoder.py)
# class Visual_Block(nn.Module): ... (as in your Encoder.py)
# Ensure Visual_Block correctly handles and outputs 5D tensors,
# preserving the Depth dimension as intended.

class visual_encoder(nn.Module):
    def __init__(self):
        super(visual_encoder, self).__init__()

        self.block1 = Visual_Block(1, 32, is_down=True)
        # Replace MaxPool3d with MaxPool2d
        # Original MaxPool3d kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)
        # This means 2D pooling on H, W dimensions with kernel 3, stride 2, padding 1
        self.pool1_2d = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.block2 = Visual_Block(32, 64)
        # Replace MaxPool3d with MaxPool2d
        self.pool2_2d = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.block3 = Visual_Block(64, 128)

        # AdaptiveMaxPool2d is for the final spatial pooling after all blocks
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))

        self.__init_weight()

    def forward(self, x):
        # Input x is expected to be 5D: (N, C_in, D, H_in, W_in)
        # e.g., (Batch, 1, Time_frames, Image_Height, Image_Width)

        # --- Block 1 & 2D Pooling 1 ---
        x = self.block1(x)  # Output x: (N, C1_out, D, H1, W1)

        N, C, D, H, W = x.shape
        # To apply MaxPool2d across H, W for each D slice:
        # Reshape x from (N, C, D, H, W) to (N*D, C, H, W)
        # by first permuting to bring N and D together.
        x_permuted = x.permute(0, 2, 1, 3, 4)  # (N, D, C, H, W)
        x_reshaped = x_permuted.reshape(N * D, C, H, W)

        x_pooled = self.pool1_2d(x_reshaped)  # Output: (N*D, C, H_new1, W_new1)

        _, _, H_new1, W_new1 = x_pooled.shape  # Get new H, W after pooling
        # Reshape back to (N, D, C, H_new1, W_new1)
        x_reshaped_back = x_pooled.reshape(N, D, C, H_new1, W_new1)
        # Permute back to (N, C, D, H_new1, W_new1)
        x = x_reshaped_back.permute(0, 2, 1, 3, 4)

        # --- Block 2 & 2D Pooling 2 ---
        x = self.block2(x)  # Output x: (N, C2_out, D, H_new1_b2, W_new1_b2)

        N, C, D, H, W = x.shape  # Update N, C, H, W if they changed in block2
        x_permuted = x.permute(0, 2, 1, 3, 4)  # (N, D, C, H, W)
        x_reshaped = x_permuted.reshape(N * D, C, H, W)

        x_pooled = self.pool2_2d(x_reshaped)  # Output: (N*D, C, H_new2, W_new2)

        _, _, H_new2, W_new2 = x_pooled.shape
        x_reshaped_back = x_pooled.reshape(N, D, C, H_new2, W_new2)
        x = x_reshaped_back.permute(0, 2, 1, 3, 4)

        # --- Block 3 ---
        x = self.block3(x)  # Output x: (N, C3_out, D, H_new2_b3, W_new2_b3)

        # --- Final Operations ---
        # Original comments: x = x.transpose(1,2) -> (B, T, C, H, W)
        # Here, N is Batch (B), D is Time (T), C is Channels (C_final)
        # Current x shape: (N, C_final, D, H_final, W_final)

        x = x.permute(0, 2, 1, 3, 4)  # (N, D, C_final, H_final, W_final)
        # This matches the (B, T, C, H, W) interpretation if N=B, D=T, C_final=C

        N_batch, D_time, C_channels, H_last, W_last = x.shape

        # Reshape for AdaptiveMaxPool2d:
        # It expects (Batch_for_2D, Channels, H, W)
        # We want to pool spatially (H_last, W_last) for each time step and batch instance.
        x = x.reshape(N_batch * D_time, C_channels, H_last, W_last)

        x = self.maxpool(x)  # self.maxpool is AdaptiveMaxPool2d((1,1))
        # Output: (N_batch*D_time, C_channels, 1, 1)

        # Reshape to (Batch, Time, Channels)
        x = x.view(N_batch, D_time, C_channels)

        return x

    def __init_weight(self):
        # Keep your original weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class audio_encoder(nn.Module):
    def __init__(self):
        super(audio_encoder, self).__init__()

        self.block1 = Audio_Block(1, 32)
        self.pool1_1d = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # self.pool1 = nn.MaxPool3d(kernel_size = (1, 1, 3), stride = (1, 1, 2), padding = (0, 0, 1))

        self.block2 = Audio_Block(32, 64)
        self.pool2_1d = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        # self.pool2 = nn.MaxPool3d(kernel_size = (1, 1, 3), stride = (1, 1, 2), padding = (0, 0, 1))

        self.block3 = Audio_Block(64, 128)

        self.__init_weight()

    def forward(self, x):
        # Input x is expected to be 4D: (N, C_in, H_in, W_in)
        # e.g., (Batch, 1, Freq_bins, Input_Time_frames)

        # --- Block 1 & 1D Pooling 1 ---
        x = self.block1(x)  # Output x: (N, C1_out, H1, W1)

        N, C, H, W = x.shape
        # To use MaxPool1d along the W (Time) dimension:
        # Reshape x from (N, C, H, W) to (N*C*H, W) to treat each "row" as a 1D signal.
        # MaxPool1d expects (Batch_for_1D, Channels_for_1D, Length_for_1D).
        # We'll make Channels_for_1D = 1.
        x_reshaped = x.reshape(N * C * H, 1, W)

        x_pooled = self.pool1_1d(x_reshaped)  # Output: (N*C*H, 1, W_new1)

        _, _, W_new1 = x_pooled.shape  # Get the new length after pooling
        # Reshape back to (N, C, H, W_new1)
        x = x_pooled.reshape(N, C, H, W_new1)

        # --- Block 2 & 1D Pooling 2 ---
        x = self.block2(x)  # Output x: (N, C2_out, H2, W_new1) (assuming H doesn't change in block2)

        N, C, H, W = x.shape  # Update N, C, H, W if they changed in block2
        x_reshaped = x.reshape(N * C * H, 1, W)

        x_pooled = self.pool2_1d(x_reshaped)  # Output: (N*C*H, 1, W_new2)

        _, _, W_new2 = x_pooled.shape
        x = x_pooled.reshape(N, C, H, W_new2)

        # --- Block 3 ---
        x = self.block3(x)  # Output x: (N, C3_out, H3, W_new2)

        # --- Final Operations ---
        # Original code: torch.mean(x, dim=2, keepdim=True)
        # dim=2 here corresponds to the H (Freq_bins) dimension.
        x = torch.mean(x, dim=2, keepdim=True)  # Output: (N, C3_out, 1, W_new2)

        # Original code: x = x.squeeze(2).transpose(1, 2)
        # .squeeze(2) -> (N, C3_out, W_new2)
        # .transpose(1, 2) -> (N, W_new2, C3_out) which is (Batch, Time_out, Features_out)
        x = x.squeeze(2).transpose(1, 2)

        return x

    def forward0(self, x):
        # x is likely (Batch, Channels, Freq_bins, Time_frames) from Audio_Block
        x = self.block1(x)

        # ---- START CORRECTION ----
        # Add a dummy depth dimension to make input 5D for MaxPool3d
        # Assuming x is (N, C, H, W), transform to (N, C, D=1, H, W)
        x = x.unsqueeze(2)
        x = self.pool1(x)  # self.pool1 is MaxPool3d
        # Remove the dummy depth dimension
        x = x.squeeze(2)
        # ---- END CORRECTION ----

        x = self.block2(x)

        # ---- START CORRECTION FOR POOL2 AS WELL ----
        x = x.unsqueeze(2)
        x = self.pool2(x)  # self.pool2 is also MaxPool3d
        x = x.squeeze(2)
        # ---- END CORRECTION FOR POOL2 ----

        x = self.block3(x)

        x = torch.mean(x, dim=2, keepdim=True)  # dim=2 here is Freq_bins (H)
        x = x.squeeze(2).transpose(1, 2)

        # x = self.block1(x)
        # x = self.pool1(x)
        #
        # x = self.block2(x)
        # x = self.pool2(x)
        #
        # x = self.block3(x)
        #
        # x = torch.mean(x, dim = 2, keepdim = True)
        # x = x.squeeze(2).transpose(1, 2)

        return x

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()