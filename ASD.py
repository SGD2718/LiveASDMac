import torch
import torch.nn as nn
import torch.nn.functional as F

import sys, time, numpy, os, subprocess, pandas, tqdm
from subprocess import PIPE

from loss import lossAV, lossV # Assuming these are in the same directory or PYTHONPATH
from model.Model import ASD_Model # Assuming this is in the same directory or PYTHONPATH

class ASD(nn.Module):
    def __init__(self, lr = 0.001, lrDecay = 0.95, **kwargs):
        super(ASD, self).__init__()

        # Determine the device
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("Using MPS device (Apple Metal).")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("Using CUDA device.")
        else:
            self.device = torch.device("cpu")
            print("Using CPU device.")

        self.model = ASD_Model().to(self.device)
        self.lossAV = lossAV().to(self.device)
        self.lossV = lossV().to(self.device)
        self.optim = torch.optim.Adam(self.parameters(), lr = lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim, step_size = 1, gamma=lrDecay)
        print(time.strftime("%m-%d %H:%M:%S") + " Model para number = %.2f"%(sum(param.numel() for param in self.model.parameters()) / 1000 / 1000))

    def train_network(self, loader, epoch, **kwargs):
        self.train()
        self.scheduler.step(epoch - 1)  # StepLR
        index, top1, lossV_val, lossAV_val, loss_val = 0, 0, 0, 0, 0 # Renamed to avoid conflict with loss module
        lr = self.optim.param_groups[0]['lr']
        r = 1.3 - 0.02 * (epoch - 1)
        for num, (audioFeature, visualFeature, labels_data) in enumerate(loader, start=1): # Renamed labels to labels_data
            self.zero_grad()

            # Move data to the determined device
            audioFeature_tensor = audioFeature[0].to(self.device)
            visualFeature_tensor = visualFeature[0].to(self.device)
            labels = labels_data[0].reshape((-1)).to(self.device) # Loss

            audioEmbed = self.model.forward_audio_frontend(audioFeature_tensor)
            visualEmbed = self.model.forward_visual_frontend(visualFeature_tensor)

            outsAV= self.model.forward_audio_visual_backend(audioEmbed, visualEmbed)
            outsV = self.model.forward_visual_backend(visualEmbed)

            nlossAV, _, _, prec = self.lossAV.forward(outsAV, labels, r)
            nlossV = self.lossV.forward(outsV, labels, r)
            nloss = nlossAV + 0.5 * nlossV

            lossV_val += nlossV.detach().cpu().numpy()
            lossAV_val += nlossAV.detach().cpu().numpy()
            loss_val += nloss.detach().cpu().numpy()
            top1 += prec.item() # Ensure prec is a scalar value if it's a tensor
            nloss.backward()
            self.optim.step()
            index += len(labels)
            sys.stderr.write(time.strftime("%m-%d %H:%M:%S") + \
            " [%2d] r: %.2f, Lr: %.5f, Training: %.2f%%, "    %(epoch, r, lr, 100 * (num / loader.__len__())) + \
            " LossV: %.5f, LossAV: %.5f, Loss: %.5f, ACC: %.2f%% \r"  %(lossV_val/(num), lossAV_val/(num), loss_val/(num), 100 * (top1/index)))
            sys.stderr.flush()

        sys.stdout.write("\n")

        return loss_val/num, lr

    def evaluate_network(self, loader, evalCsvSave, evalOrig, **kwargs):
        self.eval()
        predScores = []
        for audioFeature, visualFeature, labels_data in tqdm.tqdm(loader): # Renamed labels to labels_data
            with torch.no_grad():
                # Move data to the determined device
                audioFeature_tensor = audioFeature[0].to(self.device)
                visualFeature_tensor = visualFeature[0].to(self.device)
                labels = labels_data[0].reshape((-1)).to(self.device)

                audioEmbed  = self.model.forward_audio_frontend(audioFeature_tensor)
                visualEmbed = self.model.forward_visual_frontend(visualFeature_tensor)
                outsAV= self.model.forward_audio_visual_backend(audioEmbed, visualEmbed)
                _, predScore_tensor, _, _ = self.lossAV.forward(outsAV, labels) # Renamed predScore
                predScore = predScore_tensor[:,1].detach().cpu().numpy()
                predScores.extend(predScore)
                # break
        evalLines = open(evalOrig).read().splitlines()[1:]
        # The 'labels' variable below is for pandas Series, distinct from PyTorch tensor 'labels'
        pandas_labels = pandas.Series( ['SPEAKING_AUDIBLE' for line in evalLines])
        scores = pandas.Series(predScores)
        evalRes = pandas.read_csv(evalOrig)
        evalRes['score'] = scores
        evalRes['label'] = pandas_labels
        evalRes.drop(['label_id'], axis=1,inplace=True)
        evalRes.drop(['instance_id'], axis=1,inplace=True)
        evalRes.to_csv(evalCsvSave, index=False)
        cmd = "python -O utils/get_ava_active_speaker_performance.py -g %s -p %s "%(evalOrig, evalCsvSave)
        mAP = float(str(subprocess.run(cmd, shell=True, stdout=PIPE, stderr=PIPE).stdout).split(' ')[2][:5])
        return mAP

    def saveParameters(self, path):
        torch.save(self.state_dict(), path)

    def loadParameters(self, path):
        selfState = self.state_dict()
        # Load parameters onto the current device
        loadedState = torch.load(path, map_location=self.device)
        for name, param in loadedState.items():
            origName = name
            if name not in selfState:
                name = name.replace("module.", "")
                if name not in selfState:
                    print("%s is not in the model."%origName)
                    continue
            if selfState[name].size() != loadedState[origName].size():
                sys.stderr.write("Wrong parameter length: %s, model: %s, loaded: %s\n"%(origName, selfState[name].size(), loadedState[origName].size()))
                continue
            selfState[name].copy_(param)