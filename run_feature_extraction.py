import os
from os import path

stream = os.popen("cd /home/callbarian/C3D/videos")

for file in os.listdir("/home/callbarian/C3D/videos"):
    name , extension = path.splitext(file)
    print(name)
    os.popen('mkdir /home/callbarian/C3D/videos/' + name)
    os.popen('mv /home/callbarian/C3D/videos/' +file+ ' /home/callbarian/C3D/videos/'+name)
   
for file in os.listdir("/home/callbarian/C3D/videos"):

    for file2 in os.listdir("/home/callbarian/C3D/videos/"+file):
        save_path = "/home/callbarian/C3D/videos/" + file + '/' + file
        file_path= "/home/callbarian/C3D/videos/" + file + '/' + file2
        os.popen('ffmpeg -i ' + file_path + ' -c copy -map 0 -f segment -segment_time 60 -reset_timestamps 1 -segment_format_options movflags=+faststart ' + save_path + '%03d.mp4')
        os.popen('mv ' + file_path + ' /home/callbarian/C3D/move_videos')
       
os.popen("python extract_C3D_feature.py")
