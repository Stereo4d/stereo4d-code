<!-- omit in toc -->
Stereo4D: Learning How Things Move in 3D from Internet Stereo Videos
================================================================


<div style="line-height: 1;">
  <a href="https://stereo4d.github.io/" target="_blank" style="margin: 2px;">
    <img alt="Website" src="https://img.shields.io/badge/Website-Stereo4D-536af5?color=536af5&logoColor=white" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="https://arxiv.org/abs/2412.09621" target="_blank" style="margin: 2px;">
    <img alt="arXiv" src="https://img.shields.io/badge/arXiv-Stereo4D-red?logo=%23B31B1B" style="display: inline-block; vertical-align: middle;"/>
  </a>
</div>

arXiv 2024
<h4>

[Linyi Jin](https://jinlinyi.github.io/)<sup>1,2</sup>, 
[Richard Tucker](https://scholar.google.com/citations?user=IkpNZAoAAAAJ&hl=en)<sup>1</sup>, 
[Zhengqi Li](https://zhengqili.github.io/)<sup>1</sup>, 
[David Fouhey](https://cs.nyu.edu/~fouhey/)<sup>3</sup>, 
[Noah Snavely](https://www.cs.cornell.edu/~snavely/)<sup>1*</sup>, 
[Aleksander Hołyński](https://holynski.org/)<sup>1,4*</sup>

<sup>1</sup>Google DeepMind, <sup>2</sup>University of Michigan, <sup>3</sup>New York University, <sup>4</sup>UC Berkeley  
(*: equal contribution)
</h4>
<hr>

<p align="center">



https://github.com/user-attachments/assets/8a6c91e4-0043-4fe8-a2fe-6c14e1bc27bf




This repository contains the data processing pipeline to convert a stereoscopic
video into a dynamic point cloud, which involves stereo disparity, and 2D tracks, fusing these quantities into a consistent 3D coordinate frame, and performing
several filtering operations to ensure temporal consistent,
high-quality reconstructions.



<!-- omit in toc -->
Table of Contents
------------------- 
- [Getting Started](#getting-started)
  - [Step 0/6 Environment](#step-06-environment)
  - [Step 1/6 Obtain raw videos and camera poses](#step-16-obtain-raw-videos-and-camera-poses)
  - [Step 2/6 Rectify raw videos and convert to perspective projections](#step-26-rectify-raw-videos-and-convert-to-perspective-projections)
  - [Step 3/6 Disparity from stereo matching](#step-36-disparity-from-stereo-matching)
  - [Step 4/6 Dense point tracking](#step-46-dense-point-tracking)
  - [Step 5/6 Filter Drifting tracks](#step-56-filter-drifting-tracks)
  - [Step 6/6 Track optimization](#step-66-track-optimization)
- [Citation](#citation)
- [Acknowledgment](#acknowledgment)



## Getting Started
### Step 0/6 Environment
```bash
# Clone the Repository
git clone --recurse-submodules git@github.com:Stereo4d/stereo4d-code.git
cd stereo4d-code
git submodule update --init --recursive
cd SEA-RAFT
git apply ../sea-raft-changes.patch
cd .. 
mamba env create --file=environment.yml
```

### Step 1/6 Obtain raw videos and camera poses
The original video used as demonstration can be found at the following link:
https://www.youtube.com/watch?v=CMwZrkhQ0ck

Please obtain the video (7680 x 3840, ~805.8MB, VR180 format) and place it in the following directory:
`stereo4d_dataset/raw/CMwZrkhQ0ck.mp4`.
The VR180 format contains a side-by-side equirectangular stereo video for the left and right eyes.

The camera poses used in this project were obtained by an internal SfM pipeline.
You can find an example reference in `release_test.json`. 


### Step 2/6 Rectify raw videos and convert to perspective projections
We have observed that some VR180 videos may not be perfectly rectified. Therefore, we perform rig calibration during bundle adjustment. 
The script runs the following steps:

1.	**Extract frames** from the specified timestamps and save them as `{videoid}-raw_equirect.mp4`.

2.	**Rectify** the equirectangular video using the rig calibration result in `release_test.json` and save it as `rectified_equirect.mp4`.

3.	**Crop** the equirectangular projection to a 60° FoV perspective projection, saving the results as:

	•	`{videoid}-left_rectified.mp4` (left eye)
  
	•	`{videoid}-right_rectified.mp4` (right eye)
```bash
JAX_PLATFORMS=cpu python rectify.py \
--videoid=CMwZrkhQ0ck \
--clipid=0 \
--output_folder=stereo4d_dataset
```
Example output:

`High resolution, raw stereo video in equirectangular format.`

https://github.com/user-attachments/assets/c9f9f9ce-dcf8-4164-95e1-03d65235afb3

`512x512 60° FoV perspective video.`

https://github.com/user-attachments/assets/55976eb4-a579-4b6d-9d35-0f5b4583391e











### Step 3/6 Disparity from stereo matching
The following script loads the rectified perspective videos, calculates the disparity, and saves the results to `flows_stereo.pkl`.
We used an internal version of RAFT when developing, here we use [SEA-RAFT](https://github.com/princeton-vl/SEA-RAFT). 
We can integrate more advanced stereo methods as they become available.
```bash
python inference_raft.py \
--videoid=CMwZrkhQ0ck \
--clipid=0 \
--output_folder=stereo4d_dataset
```


Example output:

`Raw video depth from stereo matching.`

https://github.com/user-attachments/assets/be904d87-e70e-4f60-9a7f-315d51616912







### Step 4/6 Dense point tracking
We extract long-range 2D point trajectories using [BootsTAP](https://bootstap.github.io/). 
The following script runs it on perspective videos and saves results to `tapir_2d.pkl` and visualizations to `tapir_2d.mp4`. 

For every 10th frame, we uniformly initialize 128 x 128 query points on frames
of resolution 512 x 512. We then prune redundant tracks that overlap on the same pixel. 
```bash
python tracking.py \
--videoid=CMwZrkhQ0ck \
--clipid=0 \
--output_folder=stereo4d_dataset
```
Example output:

`Dense 2D tracks.`

https://github.com/user-attachments/assets/69838eb7-4dea-4fe7-85dc-90ca293f0fb5


### Step 5/6 Filter Drifting tracks
Since 2D tracks can drift on textureless regions, we discard moving 3D tracks that correspond to certain semantic categories (e.g., `walls`, `building`, `road`, `earth`, `sidewalk`), detected by DeepLabv3 on ADE20K classes.
We can integrate more advanced tracking methods as they become available.

```bash
python segmentation.py \
--videoid=CMwZrkhQ0ck \
--clipid=0 \
--output_folder=stereo4d_dataset
```
Example output:

`Dense 3D tracks projected onto video frames, without drifting tracks.`

https://github.com/user-attachments/assets/9ac253d0-d86b-482d-9a74-b0cac2bb1bd0

We then fuse these quantities into 4D reconstructions, by lifting the 2D tracks into 3D with their depth.

https://github.com/user-attachments/assets/46d436b2-3d36-42f7-b877-0fd8341b8438

Since stereo depth estimation is performed per-frame,
the initial disparity estimates (and therefore, the 3D track
positions) are likely to exhibit high-frequency temporal jitter. 

### Step 6/6 Track optimization

To ensure static points remain stationary while moving tracks maintain realistic, smooth motion, 
avoiding abrupt depth changes frame by frame, we design an optimization process (paper Eqn. 5) to get high quality 3D tracks.

```bash
python track_optimization.py \
--videoid=CMwZrkhQ0ck \
--clipid=0 \
--output_folder=stereo4d_dataset
```
Example output:

`Project the 3D tracks back to get depthmaps.`

https://github.com/user-attachments/assets/691c81b3-f13a-4ebc-ba08-1b0d684ab0be

`Final 3D tracks` (Color trails are only shown for moving points, but all points have been reconstructed in 3D).

https://github.com/user-attachments/assets/e0666abf-f8e1-4b29-a270-4d728196d966

That's it!

Citation
--------
If you find this code useful, please consider citing:

```text
@article{jin2024stereo4d,
  title={Stereo4D: Learning How Things Move in 3D from Internet Stereo Videos}, 
  author={Jin, Linyi and Tucker, Richard and Li, Zhengqi and Fouhey, David and Snavely, Noah and Holynski, Aleksander},
  journal={arXiv preprint},
  year={2024},
}
```

Acknowledgment
--------------
Thanks to Jon Barron, Ruiqi Gao, Kyle Genova, Philipp Henzler, Andrew Liu, Erika Lu, Ben Poole, Qianqian Wang, Rundi Wu, Richard Szeliski, and Stan Szymanowicz for their helpful proofreading, comments, and discussions. Thanks to Carl Doersch, Skanda Koppula, and Ignacio Rocco for their assistance with TAPVid-3D and BootsTAP. Thanks to Carlos Hernandez, Dominik Kaeser, Janne Kontkanen, Ricardo Martin-Brualla, and Changchang Wu for their help with VR180 cameras and videos.
