import os
import cv2
import numpy as np

NUM_ELEMENT = 3
TAGS = ['Normal', 'Talking', 'Yawning']
split_file = 'male_yawn_split.lst'
dataset_path = 'dataset/Male_Yawning'
save_path = 'dataset/male_split_output'

for i in range(len(TAGS)):
    path = os.path.join(save_path, TAGS[i])
    if os.path.exists(path) is False:
        os.mkdir(path)

print('split file: ', split_file)
fd = open(split_file)

for line in fd.readlines():
    line = line.strip('\n')
    line = line.split(' ')
    video_file = os.path.join(dataset_path, line[0])
    num_clips = int((len(line) - 1) / NUM_ELEMENT)
    clips_info = np.array(map(int, line[1:]))
    clips_info = clips_info.reshape(num_clips, NUM_ELEMENT)

    cap = cv2.VideoCapture(video_file)
    fcnt = cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
    fsize = (int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)))
    print(video_file)
    for i in range(num_clips):
        fstart = int(clips_info[i, 0] * fps)
        # if -1, continue go to end
        if clips_info[i, 1] > 0:
            fend = min(int(clips_info[i, 1] * fps), fcnt)
        else:
            fend = int(fcnt)
        print('%d --> %d, tag: %s' % (fstart, fend, TAGS[clips_info[i, 2]]))
        assert(fstart > 0 or fend <= fcnt)
        cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, fstart)

        [_, file_name] = os.path.split(video_file)
        [video_name, _] = os.path.splitext(file_name)
        clip_file = '%s/%s/%s-clip-%d.avi' % (save_path, TAGS[clips_info[i, 2]], video_name, i)
        writer = cv2.VideoWriter(clip_file, fourcc=cv2.cv.FOURCC('X', 'V', 'I', 'D'), fps=fps, frameSize=fsize)
        for n in range(fend-fstart):
            success, img = cap.read()
            writer.write(img)
            #cv2.imshow('video', img)
            #cv2.waitKey(40)
        writer.release()
        #cv2.waitKey(-1)
