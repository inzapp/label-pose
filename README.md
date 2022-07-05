# Label-Pose
Simple human pose estimation labeling tool

<img src="/md/label_pose.gif" width="500"><br>

## Usage
d : next image<br>
a : previous image<br>
e : next limb point<br>
q : previous limb point<br>
w : toggle to show skeleton line<br>
left click : set limb point<br>
right click : remove limb point<br>
ESC : exit program<br>

## Label format
Each point is normalized to 0 ~ 1 range value, and save it to txt format like below
```
# use_flag, x_pos, y_pos
1.0 0.508434 0.023468
1.0 0.496386 0.159061
1.0 0.214458 0.189048
1.0 0.122892 0.301173
1.0 0.154217 0.435463
1.0 0.802410 0.212516
1.0 0.860241 0.332464
1.0 0.737349 0.432855
1.0 0.318072 0.503259
1.0 0.228916 0.675359
1.0 0.349398 0.805737
1.0 0.602410 0.537158
1.0 0.544578 0.692308
1.0 0.474699 0.897001
1.0 0.469880 0.337679
0.0 0.000000 0.000000
```
