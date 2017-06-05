import numpy as np
import cv2
import sys
import os
import math

# Generate start point and label of video clips
video_root_folder = '/home/ubuser/swei/data_and_backup/dataset/YawDD_dataset/Dash/Female/'
yawn_location_file = '../data/dash_female_yawn_split.lst'
out_file = 'test.lst'

stride = 4
depth = 16
sampling_rate = 2
iou_thresh = 0.6

fd_out = open(out_file, 'w+')

fd = open(yawn_location_file, 'r')
for line in fd.readlines():
    # Read video file and yawn locations
    line = line.strip('\n')
    line = line.split(' ')
    video_file = line[0]
    locations = np.reshape(list(map(int, line[1:])), [-1, 2])

    # Get video information
    cap = cv2.VideoCapture(os.path.join(video_root_folder, video_file))
    fcnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if math.isnan(fps):
        fps = 30
    # Generate start point and label of video clips
    for idx in range(0, fcnt - (depth * sampling_rate), stride):
        sc = idx
        ec = idx + (depth * sampling_rate)
        label = 0
        for loc in locations:
            si = max(sc, loc[0]*fps)
            ei = min(ec, loc[1]*fps)
            #su = min(sc, loc[0])
            #eu = max(ec, loc[1])
            iou = float(ei - si) / float(depth * sampling_rate)
            if iou >= iou_thresh:
                label = 1
                break
        #print('%s %d %d' % (video_file, idx, label))
        #if label == 1:
        fd_out.writelines('%s %d %d\n' % (video_file, idx, label))
        fd_out.flush()

cap.release()
fd.close()
fd_out.close()

