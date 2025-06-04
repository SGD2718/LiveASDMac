import os

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'

import sys, time, tqdm, torch, argparse, glob, subprocess, warnings, cv2, pickle, numpy, pdb, math, \
    python_speech_features

from scipy import signal
from shutil import rmtree
from scipy.io import wavfile
from scipy.interpolate import interp1d
from sklearn.metrics import accuracy_score, f1_score

from scenedetect.video_manager import VideoManager
from scenedetect.scene_manager import SceneManager
from scenedetect.frame_timecode import FrameTimecode
from scenedetect.stats_manager import StatsManager
from scenedetect.detectors import ContentDetector

import utils.tools
from model.faceDetector.s3fd import S3FD
from ASD import ASD

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description="Columbia ASD Evaluation")

parser.add_argument('--videoName', type=str, default="col", help='Demo video name')
parser.add_argument('--videoFolder', type=str, default="colDataPath", help='Path for inputs, tmps and outputs')
parser.add_argument('--pretrainModel', type=str, default="weight/pretrain_AVA_CVPR.pt",
                    help='Path for the pretrained model')

parser.add_argument('--nDataLoaderThread', type=int, default=10, help='Number of workers')
parser.add_argument('--facedetScale', type=float, default=0.5,
                    help='Scale factor for face detection, the frames will be scale to 0.25 orig')
parser.add_argument('--minTrack', type=int, default=1, help='Number of min frames for each shot')
parser.add_argument('--numFailedDet', type=int, default=10,
                    help='Number of missed detections allowed before tracking is stopped')
parser.add_argument('--minFaceSize', type=int, default=10, help='Minimum face size in pixels')
parser.add_argument('--cropScale', type=float, default=0.40, help='Scale bounding box')

parser.add_argument('--start', type=int, default=0, help='The start time of the video')
parser.add_argument('--duration', type=int, default=0,
                    help='The duration of the video, when set as 0, will extract the whole video')

parser.add_argument('--evalCol', dest='evalCol', action='store_true', help='Evaluate on Columbia dataset')
parser.add_argument('--colSavePath', type=str, default="/colDataPath", help='Path for inputs, tmps and outputs')

args = parser.parse_args()

if args.evalCol == True:
    # The process is: 1. download video and labels(I have modified the format of labels to make it easiler for using)
    # 	              2. extract audio, extract video frames
    #                 3. scend detection, face detection and face tracking
    #                 4. active speaker detection for the detected face clips
    #                 5. use iou to find the identity of each face clips, compute the F1 results
    # The step 1 to 3 will take some time (That is one-time process). It depends on your cpu and gpu speed. For reference, I used 1.5 hour
    # The step 4 and 5 need less than 10 minutes
    # Need about 20G space finally
    # ```
    args.videoName = 'col'
    args.videoFolder = args.colSavePath
    args.savePath = os.path.join(args.videoFolder, args.videoName)
    args.videoPath = os.path.join(args.videoFolder, args.videoName + '.mp4')
    args.duration = 0
    if not os.path.isfile(args.videoPath):  # Download video
        link = 'https://www.youtube.com/watch?v=6GzxbrO0DHM&t=2s'
        cmd = "youtube-dl -f best -o %s '%s'" % (args.videoPath, link)
        output = subprocess.call(cmd, shell=True, stdout=None)
    if not os.path.isdir(args.videoFolder + '/col_labels'):  # Download label
        link = "1Tto5JBt6NsEOLFRWzyZEeV6kCCddc6wv"
        cmd = "gdown --id %s -O %s" % (link, args.videoFolder + '/col_labels.tar.gz')
        subprocess.call(cmd, shell=True, stdout=None)
        cmd = "tar -xzvf %s -C %s" % (args.videoFolder + '/col_labels.tar.gz', args.videoFolder)
        subprocess.call(cmd, shell=True, stdout=None)
        os.remove(args.videoFolder + '/col_labels.tar.gz')
else:
    args.videoPath = glob.glob(os.path.join(args.videoFolder, args.videoName + '.*'))[0]
    args.savePath = os.path.join(args.videoFolder, args.videoName)


def scene_detect(args):
    # CPU: Scene detection, output is the list of each shot's time duration
    videoManager = VideoManager([args.videoFilePath])
    statsManager = StatsManager()
    sceneManager = SceneManager(statsManager)
    sceneManager.add_detector(ContentDetector())
    baseTimecode = videoManager.get_base_timecode()
    videoManager.set_downscale_factor()
    videoManager.start()
    sceneManager.detect_scenes(frame_source=videoManager)
    sceneList = sceneManager.get_scene_list(baseTimecode)
    savePath = os.path.join(args.pyworkPath, 'scene.pckl')
    if sceneList == []:
        sceneList = [(videoManager.get_base_timecode(), videoManager.get_current_timecode())]
    with open(savePath, 'wb') as fil:
        pickle.dump(sceneList, fil)
        sys.stderr.write('%s - scenes detected %d\n' % (args.videoFilePath, len(sceneList)))
    return sceneList


def inference_video(args):
    # GPU: Face detection, output is the list contains the face location and score in this frame
    DET = S3FD(device='cpu')
    flist = glob.glob(os.path.join(args.pyframesPath, '*.jpg'))
    flist.sort()
    dets = []
    for fidx, fname in enumerate(flist):
        image = cv2.imread(fname)
        imageNumpy = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        bboxes = DET.detect_faces(imageNumpy, conf_th=0.8, scales=[0.5])
        dets.append([])
        for bbox in bboxes:
            dets[-1].append({'frame': fidx, 'bbox': (bbox[:-1]).tolist(),
                             'conf': bbox[-1]})  # dets has the frames info, bbox info, conf info
        sys.stderr.write('%s-%05d; %d dets\r' % (args.videoFilePath, fidx, len(dets[-1])))
    savePath = os.path.join(args.pyworkPath, 'faces.pckl')
    with open(savePath, 'wb') as fil:
        pickle.dump(dets, fil)
    return dets


def bb_intersection_over_union(boxA, boxB, evalCol=False):
    # CPU: IOU Function to calculate overlap between two image
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Compute the area of intersection rectangle
    interArea = max(0, xB - xA) * max(0, yB - yA)

    # Compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    if evalCol:
        # Specific IOU for Columbia evaluation: intersection over area of boxA
        if float(boxAArea) == 0:
            # If boxA has no area, IOU is 0 to avoid division by zero.
            return 0.0
        iou = interArea / float(boxAArea)
    else:
        # Standard IOU: intersection over union
        # Union = AreaA + AreaB - InterArea
        unionArea = float(boxAArea + boxBArea - interArea)

        if unionArea == 0:
            # If the union area is 0 (e.g., both boxes have zero area),
            # then IOU is 0.
            return 0.0
        iou = interArea / unionArea
    return iou


def track_shot(args, sceneFaces):
    # CPU: Face tracking
    iouThres = 0.5  # Minimum IOU between consecutive face detections
    tracks = []
    while True:
        track = []
        for frameFaces in sceneFaces:
            for face in frameFaces:
                if track == []:
                    track.append(face)
                    frameFaces.remove(face)
                elif face['frame'] - track[-1]['frame'] <= args.numFailedDet:
                    iou = bb_intersection_over_union(face['bbox'], track[-1]['bbox'])
                    if iou > iouThres:
                        track.append(face)
                        frameFaces.remove(face)
                        continue
                else:
                    break
        if track == []:
            break
        elif len(track) > args.minTrack:
            frameNum = numpy.array([f['frame'] for f in track])
            bboxes = numpy.array([numpy.array(f['bbox']) for f in track])
            frameI = numpy.arange(frameNum[0], frameNum[-1] + 1)
            bboxesI = []
            for ij in range(0, 4):
                interpfn = interp1d(frameNum, bboxes[:, ij])
                bboxesI.append(interpfn(frameI))
            bboxesI = numpy.stack(bboxesI, axis=1)
            if max(numpy.mean(bboxesI[:, 2] - bboxesI[:, 0]),
                   numpy.mean(bboxesI[:, 3] - bboxesI[:, 1])) > args.minFaceSize:
                tracks.append({'frame': frameI, 'bbox': bboxesI})
    return tracks


def crop_video(args, track, cropFile):
    # CPU: crop the face clips
    print(f"track={track}")
    flist = glob.glob(os.path.join(args.pyframesPath, '*.jpg'))  # Read the frames
    flist.sort()
    vOut = cv2.VideoWriter(cropFile + 't.avi', cv2.VideoWriter_fourcc(*'XVID'), 25, (224, 224))  # Write video
    dets = {'x': [], 'y': [], 's': []}
    for det in track['bbox']:  # Read the tracks
        dets['s'].append(max((det[3] - det[1]), (det[2] - det[0])) / 2)
        dets['y'].append((det[1] + det[3]) / 2)  # crop center x
        dets['x'].append((det[0] + det[2]) / 2)  # crop center y
    dets['s'] = signal.medfilt(dets['s'], kernel_size=13)  # Smooth detections
    dets['x'] = signal.medfilt(dets['x'], kernel_size=13)
    dets['y'] = signal.medfilt(dets['y'], kernel_size=13)
    for fidx, frame in enumerate(track['frame']):
        cs = args.cropScale
        bs = dets['s'][fidx]  # Detection box size
        bsi = int(bs * (1 + 2 * cs))  # Pad videos by this amount
        image = cv2.imread(flist[frame])
        frame = numpy.pad(image, ((bsi, bsi), (bsi, bsi), (0, 0)), 'constant', constant_values=(110, 110))
        my = dets['y'][fidx] + bsi  # BBox center Y
        mx = dets['x'][fidx] + bsi  # BBox center X
        face = frame[int(my - bs):int(my + bs * (1 + 2 * cs)), int(mx - bs * (1 + cs)):int(mx + bs * (1 + cs))]
        if face.shape[0] < 1 or face.shape[1] < 1:  # If the face is not detected, skip
            continue
        vOut.write(cv2.resize(face, (224, 224)))
    audioTmp = cropFile + '.wav'
    audioStart = (track['frame'][0]) / 25
    audioEnd = (track['frame'][-1] + 1) / 25
    vOut.release()
    command = (
            "ffmpeg -y -i %s -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 -threads %d -ss %.3f -to %.3f %s -loglevel panic" % \
            (args.audioFilePath, args.nDataLoaderThread, audioStart, audioEnd, audioTmp))
    output = subprocess.call(command, shell=True, stdout=None)  # Crop audio file
    _, audio = wavfile.read(audioTmp)
    command = ("ffmpeg -y -i %st.avi -i %s -threads %d -c:v copy -c:a copy %s.avi -loglevel panic" % \
               (cropFile, audioTmp, args.nDataLoaderThread, cropFile))  # Combine audio and video file
    output = subprocess.call(command, shell=True, stdout=None)
    os.remove(cropFile + 't.avi')
    return {'track': track, 'proc_track': dets}


def extract_MFCC(file, outPath):
    # CPU: extract mfcc
    sr, audio = wavfile.read(file)
    mfcc = python_speech_features.mfcc(audio, sr)  # (N_frames, 13)   [1s = 100 frames]
    featuresPath = os.path.join(outPath, file.split('/')[-1].replace('.wav', '.npy'))
    numpy.save(featuresPath, mfcc)


def evaluate_network(files, args):
    # GPU: active speaker detection by pretrained model
    s = ASD()
    device = torch.device('mps')
    s.loadParameters(args.pretrainModel)
    sys.stderr.write("Model %s loaded from previous state! \r\n" % args.pretrainModel)
    s.eval()
    allScores = []
    # durationSet = {1,2,4,6} # To make the result more reliable
    durationSet = {1, 1, 1, 2, 2, 2, 3, 3, 4, 5, 6}  # Use this line can get more reliable result
    for file in tqdm.tqdm(files, total=len(files)):
        fileName = os.path.splitext(file.split('/')[-1])[0]  # Load audio and video
        _, audio = wavfile.read(os.path.join(args.pycropPath, fileName + '.wav'))
        audioFeature = python_speech_features.mfcc(audio, 16000, numcep=13, winlen=0.025, winstep=0.010)
        video = cv2.VideoCapture(os.path.join(args.pycropPath, fileName + '.avi'))
        videoFeature_list = []
        i = 0
        while video.isOpened():
            print(f'Frame {i}')
            i+=1
            ret, frameData = video.read()
            if ret:
                face = cv2.cvtColor(frameData, cv2.COLOR_BGR2GRAY)
                face = cv2.resize(face, (224, 224))
                # Center crop 112x112
                h_center, w_center = 112, 112
                crop_size = 112
                face = face[h_center - crop_size // 2: h_center + crop_size // 2,
                            w_center - crop_size // 2: w_center + crop_size // 2]
                videoFeature_list.append(face)
            else:
                break
        video.release()

        if not videoFeature_list:
            print(f"Warning: No video frames extracted from 0003.mp4. Skipping.")
            allScores.append(numpy.array([]))  # Append empty score or handle as needed
            continue

        videoFeature = numpy.array(videoFeature_list)
        print(f"  Extracted audioFeature shape: {audioFeature.shape}, videoFeature shape: {videoFeature.shape}")

        # Ensure videoFeature is 3D (Frames, Height, Width)
        if videoFeature.ndim != 3 or videoFeature.shape[0] == 0:
            print(
                f"Warning: videoFeature for {fileName} is not valid 3D or is empty. Shape: {videoFeature.shape}. Skipping.")
            allScores.append(numpy.array([]))
            continue

        # videoFeature = numpy.array(videoFeature)
        length = min((audioFeature.shape[0] - audioFeature.shape[0] % 4) / 100, videoFeature.shape[0])
        audioFeature = audioFeature[:int(round(length * 100)), :]
        videoFeature = videoFeature[:int(round(length * 25)), :, :]
        allScore = []  # Evaluation use model
        for duration in durationSet:
            batchSize = int(math.ceil(length / duration))
            scores = []
            with torch.no_grad():
                for i in range(batchSize):
                    inputA = torch.FloatTensor(audioFeature[i * duration * 100:(i + 1) * duration * 100, :]).unsqueeze(
                        0).to(device)
                    inputV = torch.FloatTensor(
                        videoFeature[i * duration * 25: (i + 1) * duration * 25, :, :]).unsqueeze(0).to(device)
                    embedA = s.model.forward_audio_frontend(inputA)
                    embedV = s.model.forward_visual_frontend(inputV)
                    out = s.model.forward_audio_visual_backend(embedA, embedV)
                    score = s.lossAV.forward(out, labels=None)
                    scores.extend(score)
            allScore.append(scores)
        allScore = numpy.round((numpy.mean(numpy.array(allScore), axis=0)), 1).astype(float)
        allScores.append(allScore)
    return allScores


# Assuming utils.tools is available in the path
# import utils.tools # This line would be in the main script, not necessarily repeated here

def visualization(tracks, scores, args):
    print("\033[1;32m--- Starting Visualization ---")
    print(f"Number of tracks received: {len(tracks)}")
    print(f"Length of scores array received: {len(scores)}")

    if not tracks or not scores:
        print("Warning: Tracks or scores are empty. No visualization will be generated.")
        # Early exit or handle as appropriate if there's no data
        # For now, let it proceed to see if flist causes issues later
    elif len(tracks) != len(scores):
        print(
            f"Warning: Mismatch between number of tracks ({len(tracks)}) and scores ({len(scores)}). This might lead to errors.")

    # CPU: visulize the result for video format
    flist = glob.glob(os.path.join(args.pyframesPath, '*.jpg'))
    flist.sort()

    if not flist:
        print(f"Error: No JPEG frames found in {args.pyframesPath}. Cannot proceed with visualization.")
        return  # Cannot continue if there are no frames

    print(f"Found {len(flist)} frames for visualization in {args.pyframesPath}")

    faces = [[] for _ in range(len(flist))]  # Use _ if i is not used
    print(f"Initialized 'faces' list with {len(faces)} empty sublists (one per frame).")

    for tidx, track in enumerate(tracks):
        if tidx < len(scores):  # Check to prevent index out of bounds if scores is shorter
            score = scores[tidx]
            print(f"\nProcessing Track {tidx + 1}/{len(tracks)}:")
            print(f"  Number of frames in this track: {len(track['track']['frame'])}")
            print(f"  Number of scores for this track: {len(score)}")
            # print(f"  Scores for this track: {score}") # Potentially very verbose

            for fidx, frame_num in enumerate(
                    track['track']['frame'].tolist()):  # Renamed 'frame' to 'frame_num' for clarity
                if fidx < len(score):  # Check to prevent index out of bounds for score array
                    # average smoothing
                    s_raw = score[max(fidx - 2, 0): min(fidx + 3, len(score))]
                    s_mean = numpy.mean(s_raw)

                    if frame_num < len(faces):  # Ensure frame_num is a valid index for 'faces'
                        face_data = {
                            'track': tidx,
                            'score': float(s_mean),
                            's': track['proc_track']['s'][fidx],  # 's' here is size
                            'x': track['proc_track']['x'][fidx],
                            'y': track['proc_track']['y'][fidx]
                        }
                        faces[frame_num].append(face_data)
                        # This can be very verbose, enable if needed for detailed debugging of this part
                        # print(f"  Appended to faces[{frame_num}]: {face_data}")
                    else:
                        print(
                            f"  Warning: frame_num {frame_num} is out of bounds for 'faces' list (size {len(faces)}). Skipping face data for this frame.")
                else:
                    print(
                        f"  Warning: fidx {fidx} is out of bounds for score array (length {len(score)}) in track {tidx}. Skipping score processing.")
        else:
            print(
                f"  Warning: tidx {tidx} is out of bounds for scores array (length {len(scores)}). Skipping track processing.")

    print(f"\nLoading first image for dimensions: {flist[0]}")
    firstImage = cv2.imread(flist[0])
    if firstImage is None:
        print(f"Error: Could not read the first image {flist[0]}.")
        return

    fh, fw = firstImage.shape[:2]  # More robust way to get height and width
    print(f"Video dimensions: Width={fw}, Height={fh}")

    output_video_only_path = os.path.join(args.pyaviPath, 'video_only.avi')
    print(f"Initializing VideoWriter for: {output_video_only_path}")
    vOut = cv2.VideoWriter(output_video_only_path, cv2.VideoWriter_fourcc(*'XVID'), 25,
                           (fw, fh))

    if not vOut.isOpened():
        print(
            f"Error: VideoWriter could not be opened for {output_video_only_path}. Check FFmpeg/codec availability and path permissions.")
        return

    colorDict = {0: 0, 1: 255}  # 0 for red (non-active), 1 for green (active)
    print("\nStarting frame-by-frame processing for drawing...")

    for fidx, fname in tqdm.tqdm(enumerate(flist), total=len(flist), desc="Processing frames"):
        image = cv2.imread(fname)
        if image is None:
            print(f"Warning: Could not read frame {fname}. Skipping.")
            # Create a black frame to keep video length consistent, or skip
            image = numpy.zeros((fh, fw, 3), dtype=numpy.uint8)
            # continue

        if fidx < len(faces) and faces[fidx]:  # Check if there are any faces detected for this frame
            # print(f"Frame {fidx}: Processing {len(faces[fidx])} detected faces.") # Can be verbose
            for face_info in faces[fidx]:
                score_val = face_info['score']
                is_active_speaker = score_val >= 0

                # Determine color: 0 for Red (Non-active), 1 for Green (Active)
                # color_key will be 1 if active (score >= 0), 0 otherwise.
                color_key = int(is_active_speaker)
                clr_component = colorDict[color_key]  # This will be 255 for green, 0 for red

                # Box color: (B, G, R)
                # Green: (0, 255, 0) -> clr_component = 255
                # Red:   (0, 0, 255) -> clr_component = 0
                box_color = (0, clr_component, 255 - clr_component)

                text_to_display = f"{score_val:.1f}"  # Format score to one decimal place

                # Face bounding box coordinates
                center_x = int(face_info['x'])
                center_y = int(face_info['y'])
                size = int(face_info['s'])  # This 's' is half the side length of the square box around the center

                x1 = center_x - size
                y1 = center_y - size
                x2 = center_x + size
                y2 = center_y + size

                print(
                    f"  Frame {fidx}, Face Track {face_info['track']}: Score={score_val:.2f}, Active={is_active_speaker}, ColorComponent={clr_component}, BoxColor={box_color}")
                print(f"    Drawing rectangle at ({x1},{y1}) to ({x2},{y2})")

                cv2.rectangle(image, (x1, y1), (x2, y2), box_color, 10)
                cv2.putText(image, text_to_display, (x1, y1 - 10),  # Adjusted text position slightly
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, box_color, 5)
        # else: # This can be very verbose if many frames have no faces
        # print(f"Frame {fidx}: No faces to draw.")

        vOut.write(image)

    print("\nReleasing VideoWriter.")
    vOut.release()

    print("--- Video Processing (FFmpeg) ---")
    video_only_avi = os.path.join(args.pyaviPath, 'video_only.avi')
    audio_wav = os.path.join(args.pyaviPath, 'audio.wav')
    video_out_avi = os.path.join(args.pyaviPath, 'video_out.avi')

    command = (f"ffmpeg -y -i \"{video_only_avi}\" -i \"{audio_wav}\" "
               f"-threads {args.nDataLoaderThread} -c:v copy -c:a copy \"{video_out_avi}\" -loglevel panic")
    print(f"Executing FFmpeg command: {command}")
    try:
        subprocess.call(command, shell=True, stdout=None)  # Consider capturing output for debugging
        print(f"FFmpeg command executed. Output (if any) not captured by default.")
    except Exception as e:
        print(f"Error during FFmpeg command execution: {e}")

    # The following conversion calls are from your provided code
    # Assuming utils.tools.convert_avi_to_mp4 is defined elsewhere
    print("\n--- AVI to MP4 Conversion (as per user code) ---")

    def robust_convert_avi_to_mp4(avi_path, mp4_path):
        print(f"Attempting to convert {avi_path} to {mp4_path}")
        if not os.path.exists(avi_path):
            print(f"  Error: Input AVI file not found: {avi_path}")
            return
        try:
            # This is a placeholder if utils.tools is not directly usable here
            # Replace with actual conversion logic if needed, or ensure utils.tools is importable
            # For now, just simulating the call for structure
            # utils.tools.convert_avi_to_mp4(avi_path, mp4_path)

            # Using subprocess for a direct FFmpeg call as an example of conversion
            # if utils.tools.convert_avi_to_mp4 is not defined in this scope
            conversion_command = [
                'ffmpeg', '-y', '-i', avi_path,
                '-c:v', 'libx264', '-preset', 'medium', '-crf', '23',
                '-c:a', 'aac', '-b:a', '128k',
                '-movflags', '+faststart',
                mp4_path, '-loglevel', 'error'  # Show only errors
            ]
            print(f"  Executing conversion: {' '.join(conversion_command)}")
            result = subprocess.run(conversion_command, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"  Successfully converted to {mp4_path}")
            else:
                print(f"  Error converting {avi_path} to {mp4_path}:")
                print(f"    FFmpeg stderr: {result.stderr}")

        except Exception as e:
            print(f"  Exception during conversion of {avi_path}: {e}")

    # Paths for conversion
    video_avi = os.path.join(args.pyaviPath, 'video.avi')  # Original video
    video_only_mp4 = os.path.join(args.pyaviPath, 'video_only.mp4')
    video_out_mp4 = os.path.join(args.pyaviPath, 'video_out.mp4')
    video_mp4 = os.path.join(args.pyaviPath, 'video.mp4')  # Converted original video

    robust_convert_avi_to_mp4(video_only_avi, video_only_mp4)
    robust_convert_avi_to_mp4(video_out_avi, video_out_mp4)
    robust_convert_avi_to_mp4(video_avi, video_mp4)  # Converting the original input .avi as well

    print("--- Visualization and Conversion Complete ---\033[0m")


def evaluate_col_ASD(tracks, scores, args):
    txtPath = args.videoFolder + '/col_labels/fusion/*.txt'  # Load labels
    predictionSet = {}
    for name in {'long', 'bell', 'boll', 'lieb', 'sick', 'abbas'}:
        predictionSet[name] = [[], []]
    dictGT = {}
    txtFiles = glob.glob("%s" % txtPath)
    for file in txtFiles:
        lines = open(file).read().splitlines()
        idName = file.split('/')[-1][:-4]
        for line in lines:
            data = line.split('\t')
            frame = int(int(data[0]) / 29.97 * 25)
            x1 = int(data[1])
            y1 = int(data[2])
            x2 = int(data[1]) + int(data[3])
            y2 = int(data[2]) + int(data[3])
            gt = int(data[4])
            if frame in dictGT:
                dictGT[frame].append([x1, y1, x2, y2, gt, idName])
            else:
                dictGT[frame] = [[x1, y1, x2, y2, gt, idName]]
    flist = glob.glob(os.path.join(args.pyframesPath, '*.jpg'))  # Load files
    flist.sort()
    faces = [[] for i in range(len(flist))]
    for tidx, track in enumerate(tracks):
        score = scores[tidx]
        for fidx, frame in enumerate(track['track']['frame'].tolist()):
            s = numpy.mean(score[max(fidx - 2, 0): min(fidx + 3, len(score) - 1)])  # average smoothing
            faces[frame].append({'track': tidx, 'score': float(s), 's': track['proc_track']['s'][fidx],
                                 'x': track['proc_track']['x'][fidx], 'y': track['proc_track']['y'][fidx]})
    for fidx, fname in tqdm.tqdm(enumerate(flist), total=len(flist)):
        if fidx in dictGT:  # This frame has label
            for gtThisFrame in dictGT[fidx]:  # What this label is ?
                faceGT = gtThisFrame[0:4]
                labelGT = gtThisFrame[4]
                idGT = gtThisFrame[5]
                ious = []
                for face in faces[fidx]:  # Find the right face in my result
                    faceLocation = [int(face['x'] - face['s']), int(face['y'] - face['s']), int(face['x'] + face['s']),
                                    int(face['y'] + face['s'])]
                    faceLocation_new = [int(face['x'] - face['s']) // 2, int(face['y'] - face['s']) // 2,
                                        int(face['x'] + face['s']) // 2, int(face['y'] + face['s']) // 2]
                    iou = bb_intersection_over_union(faceLocation_new, faceGT, evalCol=True)
                    if iou > 0.5:
                        ious.append([iou, round(face['score'], 2)])
                if len(ious) > 0:  # Find my result
                    ious.sort()
                    labelPredict = ious[-1][1]
                else:
                    labelPredict = 0
                x1 = faceGT[0]
                y1 = faceGT[1]
                width = faceGT[2] - faceGT[0]
                predictionSet[idGT][0].append(labelPredict)
                predictionSet[idGT][1].append(labelGT)
    names = ['long', 'bell', 'boll', 'lieb', 'sick', 'abbas']  # Evaluate
    names.sort()
    F1s = 0
    for i in names:
        scores = numpy.array(predictionSet[i][0])
        labels = numpy.array(predictionSet[i][1])
        scores = numpy.int64(scores > 0)
        F1 = f1_score(labels, scores)
        ACC = accuracy_score(labels, scores)
        if i != 'abbas':
            F1s += F1
            print("%s, ACC:%.2f, F1:%.2f" % (i, 100 * ACC, 100 * F1))
    print("Average F1:%.2f" % (100 * (F1s / 5)))


# Main function
def main():
    print('\n\033[1;34m------------------------------ RUNNING -------------------------------\033[0m\n')
    # This preprocesstion is modified based on this [repository](https://github.com/joonson/syncnet_python).
    # ```
    # .
    # ├── pyavi
    # │         ├── audio.wav (Audio from input video)
    # │         ├── video.avi (Copy of the input video)
    # │         ├── video_only.avi (Output video without audio)
    # │         └── video_out.avi  (Output video with audio)
    # ├── pycrop (The detected face videos and audios)
    # │   ├── 000000.avi
    # │   ├── 000000.wav
    # │   ├── 000001.avi
    # │   ├── 000001.wav
    # │   └── ...
    # ├── pyframes (All the video frames in this video)
    # │   ├── 000001.jpg
    # │   ├── 000002.jpg
    # │   └── ...
    # └── pywork
    #     ├── faces.pckl (face detection result)
    #     ├── scene.pckl (scene detection result)
    #     ├── scores.pckl (ASD result)
    #     └── tracks.pckl (face tracking result)
    # ```

    # Initialization
    args.pyaviPath = os.path.join(args.savePath, 'pyavi')
    args.pyframesPath = os.path.join(args.savePath, 'pyframes')
    args.pyworkPath = os.path.join(args.savePath, 'pywork')
    args.pycropPath = os.path.join(args.savePath, 'pycrop')
    if os.path.exists(args.savePath):
        rmtree(args.savePath)
    os.makedirs(args.pyaviPath, exist_ok=True)  # The path for the input video, input audio, output video
    os.makedirs(args.pyframesPath, exist_ok=True)  # Save all the video frames
    os.makedirs(args.pyworkPath, exist_ok=True)  # Save the results in this process by the pckl method
    os.makedirs(args.pycropPath, exist_ok=True)  # Save the detected face clips (audio+video) in this process

    # Extract video
    args.videoFilePath = os.path.join(args.pyaviPath, 'video.avi')
    # If duration did not set, extract the whole video, otherwise extract the video from 'args.start' to 'args.start + args.duration'
    if args.duration == 0:
        command = ("ffmpeg -y -i %s -qscale:v 2 -threads %d -async 1 -r 25 %s -loglevel panic" % \
                   (args.videoPath, args.nDataLoaderThread, args.videoFilePath))
    else:
        command = ("ffmpeg -y -i %s -qscale:v 2 -threads %d -ss %.3f -to %.3f -async 1 -r 25 %s -loglevel panic" % \
                   (args.videoPath, args.nDataLoaderThread, args.start, args.start + args.duration, args.videoFilePath))
    subprocess.call(command, shell=True, stdout=None)
    sys.stderr.write(
        time.strftime("%Y-%m-%d %H:%M:%S") + " Extract the video and save in %s \r\n" % (args.videoFilePath))

    # Extract audio
    args.audioFilePath = os.path.join(args.pyaviPath, 'audio.wav')
    command = ("ffmpeg -y -i %s -qscale:a 0 -ac 1 -vn -threads %d -ar 16000 %s -loglevel panic" % \
               (args.videoFilePath, args.nDataLoaderThread, args.audioFilePath))
    subprocess.call(command, shell=True, stdout=None)
    sys.stderr.write(
        time.strftime("%Y-%m-%d %H:%M:%S") + " Extract the audio and save in %s \r\n" % (args.audioFilePath))

    # Extract the video frames
    command = ("ffmpeg -y -i %s -qscale:v 2 -threads %d -f image2 %s -loglevel panic" % \
               (args.videoFilePath, args.nDataLoaderThread, os.path.join(args.pyframesPath, '%06d.jpg')))
    subprocess.call(command, shell=True, stdout=None)
    sys.stderr.write(
        time.strftime("%Y-%m-%d %H:%M:%S") + " Extract the frames and save in %s \r\n" % (args.pyframesPath))

    # Scene detection for the video frames
    scene = scene_detect(args)
    sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Scene detection and save in %s \r\n" % (args.pyworkPath))

    # Face detection for the video frames
    faces = inference_video(args)
    sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Face detection and save in %s \r\n" % (args.pyworkPath))

    # Face tracking
    allTracks, vidTracks = [], []
    for shot in scene:
        if shot[1].frame_num - shot[0].frame_num >= args.minTrack:  # Discard the shot frames less than minTrack frames
            allTracks.extend(track_shot(args, faces[shot[0].frame_num:shot[
                1].frame_num]))  # 'frames' to present this tracks' timestep, 'bbox' presents the location of the faces
    sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Face track and detected %d tracks \r\n" % len(allTracks))
    print("Tracks=", allTracks)

    # Face clips cropping
    for ii, track in tqdm.tqdm(enumerate(allTracks), total=len(allTracks)):
        vidTracks.append(crop_video(args, track, os.path.join(args.pycropPath, '%05d' % ii)))
    savePath = os.path.join(args.pyworkPath, 'tracks.pckl')
    with open(savePath, 'wb') as fil:
        pickle.dump(vidTracks, fil)
    sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Face Crop and saved in %s tracks \r\n" % args.pycropPath)
    fil = open(savePath, 'rb')
    vidTracks = pickle.load(fil)

    # Active Speaker Detection
    files = glob.glob("%s/*.avi" % args.pycropPath)
    files.sort()
    scores = evaluate_network(files, args)
    savePath = os.path.join(args.pyworkPath, 'scores.pckl')
    with open(savePath, 'wb') as fil:
        pickle.dump(scores, fil)
    sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Scores extracted and saved in %s \r\n" % args.pyworkPath)

    if args.evalCol == True:
        evaluate_col_ASD(vidTracks, scores,
                         args)  # The columnbia video is too big for visualization. You can still add the `visualization` funcition here if you want
        quit()
    else:
        # Visualization, save the result as the new video
        visualization(vidTracks, scores, args)


if __name__ == '__main__':
    main()
    x = 0/0
    model_wrapper = ASD()
    model_wrapper.loadParameters("weight/pretrain_AVA_CVPR.pt")
    model = model_wrapper.model
    #model.load_state_dict(torch.load('weight/pretrain_AVA_CVPR.pt'))
    model_wrapper.eval()

    duration = 5
    audio_input = torch.randn(1, duration * 100, 13).to(torch.device('mps'))
    visual_input = torch.randn(1, duration * 25, 112, 112).to(torch.device('mps'))

    # Trace the entire submodule with named inputs
    traced_audio = torch.jit.trace(
        model_wrapper.model,
        {'forward_audio_frontend': audio_input}
    )

    traced_visual = torch.jit.trace(
        model_wrapper.model,
        {'forward_visual_frontend': visual_input}
    )

    # Trace backend (joint AV)
    with torch.no_grad():
        embedA = model_wrapper.model.forward_audio_frontend(audio_input)
        embedV = model_wrapper.model.forward_visual_frontend(visual_input)

    traced_backend = torch.jit.trace(
        model_wrapper.model,
        {'forward_audio_visual_backend': (embedA, embedV)}
    )

    # ---- Convert to Core ML ----
    import coremltools as ct

    print("Converting audio frontend...")
    mlmodel_audio = ct.convert(
        traced_audio,
        inputs=[ct.TensorType(name="audio_input", shape=audio_input.shape)],
        compute_units=ct.ComputeUnit.ALL
    )
    mlmodel_audio.save("coreml_models/audio_frontend.mlmodel")

    print("Converting visual frontend...")
    mlmodel_visual = ct.convert(
        traced_visual,
        inputs=[ct.TensorType(name="video_input", shape=visual_input.shape)],
        compute_units=ct.ComputeUnit.ALL
    )
    mlmodel_visual.save("coreml_models/visual_frontend.mlmodel")

    print("Converting audio-visual backend...")
    mlmodel_backend = ct.convert(
        traced_backend,
        inputs=[
            ct.TensorType(name="audio_embedding", shape=embedA.shape),
            ct.TensorType(name="visual_embedding", shape=embedV.shape)
        ],
        compute_units=ct.ComputeUnit.ALL
    )
    mlmodel_backend.save("coreml_models/audio_visual_backend.mlmodel")

    print("✅ Done. Saved as:")
    print(" - audio_frontend.mlmodel")
    print(" - visual_frontend.mlmodel")
    print(" - audio_visual_backend.mlmodel")
