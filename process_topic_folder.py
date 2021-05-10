import json
import os
import shutil

f = open('raw/WLASL.json')
action_json = json.load(f)

for action in action_json:
    category = action['gloss']
    item = 0
    for instance in action['instances']:
        video_id = instance['video_id']
        frame_end = instance['frame_end']
        frame_start = instance['frame_start']
        if frame_end != -1 or frame_start != 1:
            print(f'Not supported multi action in one video with video id {video_id}')
            continue
        if not os.path.exists(f'raw/processed/{category}'):
            os.mkdir(f'raw/processed/{category}')
        if not os.path.exists(f'raw/raw_videos_mp4/{video_id}.mp4'):
            continue
        shutil.copyfile(f'raw/raw_videos_mp4/{video_id}.mp4', f'raw/processed/{category}/{video_id}.mp4')
        item += 1
    if item == 0:
        os.remove(f'raw/processed/{category}')
