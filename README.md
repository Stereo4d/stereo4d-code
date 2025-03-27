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

CVPR 2025
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





https://github.com/user-attachments/assets/45f1f704-7962-4411-981c-2dd012d73b4c








This repository contains the data processing pipeline to convert a stereoscopic
video into a dynamic point cloud, which involves stereo disparity, and 2D tracks, fusing these quantities into a consistent 3D coordinate frame, and performing
several filtering operations to ensure temporal consistent,
high-quality reconstructions.

<em>This is not an officially supported Google product.</em>


<!-- omit in toc -->
Table of Contents
------------------- 
  - [Getting Started](#getting-started)
    - [Step 0/6 Environment](#step-06-environment)
    - [Step 1/6 Download Stereo4D dataset](#step-16-download-stereo4d-dataset)
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

### Step 1/6 Download Stereo4D dataset
We have released Stereo4D dataset annotations (4.3 TB) on Google Storage Bucket.
https://console.cloud.google.com/storage/browser/stereo4d/.
The annotations are under [CC license](https://creativecommons.org/licenses/by/4.0/legalcode.txt).  

For each video clip, we release:
```
{
  'name': clip unique id <video_id>_<first_frame_time_stamp>,
  'video_id': the link to the video `https://www.youtube.com/watch?v=<video_id>,
  'timestamps': a list of frame time stamp from the original video
  'camera2world': a list of camera poses corresponding to the rectified frames.
  'track_lengths', 'track_indices', 'track_coordinates': 3D tracks, will be loaded by utils/load_dataset_npz()
  'rectified2rig': rotation matrix used to rectify frames.
  'fov_bounds': camera intrinsics of the VR180 frame, which will be used to get perspective frames..
}
``` 

Please follow [gcloud installation guidance](https://cloud.google.com/sdk/docs/install-sdk#installing_the_latest_version) to download the npz files, or

```bash
# Install gcloud sdk
curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-linux-x86_64.tar.gz
tar -xf google-cloud-cli-linux-x86_64.tar.gz
./google-cloud-sdk/install.sh
./google-cloud-sdk/bin/gcloud init
```


```bash
# To download one example
mkdir -p stereo4d_dataset/npz
gcloud storage cp gs://stereo4d/train/CMwZrkhQ0ck_130030030.npz stereo4d_dataset/npz
``` 

```bash 
# To download full dataset
gsutil -m cp -R gs://stereo4d .
```


### Step 2/6 Rectify raw videos and convert to perspective projections
We have observed that some VR180 videos may not be perfectly rectified. Therefore, we perform rig calibration during bundle adjustment. 
The script runs the following steps:

1.	**Extract frames** from the specified `timestamps` and save them as `{videoid}-raw_equirect.mp4`.

2.	**Rectify** the equirectangular video using the rig calibration result in `rectified2rig` and save it as `rectified_equirect.mp4`.

3.	**Crop** the equirectangular projection to a 60° FoV perspective projection, saving the results as:

	•	`{videoid}-left_rectified.mp4` (left eye)
  
	•	`{videoid}-right_rectified.mp4` (right eye)
```bash
JAX_PLATFORMS=cpu python rectify.py \
--vid=CMwZrkhQ0ck_130030030
```
Example output:

`High resolution, raw stereo video in equirectangular format.`




https://github.com/user-attachments/assets/36fd5958-2423-49a3-bb3c-0f19a94030f9




`512x512 60° FoV perspective video.`




https://github.com/user-attachments/assets/927f60b1-a492-4e61-b3c6-ed7e86671333




🎉 The released `.npz` files already contain 3D tracks, you can skip the remaining steps and directly use example to visualize them.

[Notebook for visualization](./track_visualization.ipynb)


**If you want to reproduce the 3D tracks, continue with the following steps.**

### Step 3/6 Disparity from stereo matching
The following script loads the rectified perspective videos, calculates the disparity, and saves the results to `flows_stereo.pkl`.
We used an internal version of RAFT when developing, here we use [SEA-RAFT](https://github.com/princeton-vl/SEA-RAFT). 
We can integrate more advanced stereo methods as they become available.
```bash
python inference_raft.py \
--vid=CMwZrkhQ0ck_130030030
```


Example output:

`Raw video depth from stereo matching.`




https://github.com/user-attachments/assets/d4959124-d357-4220-8822-1bf1b8de900c








### Step 4/6 Dense point tracking
We extract long-range 2D point trajectories using [BootsTAP](https://bootstap.github.io/). 
The following script runs it on perspective videos and saves results to `tapir_2d.pkl` and visualizations to `tapir_2d.mp4`. 

For every 10th frame, we uniformly initialize 128 x 128 query points on frames
of resolution 512 x 512. We then prune redundant tracks that overlap on the same pixel. 
```bash
python tracking.py \
--vid=CMwZrkhQ0ck_130030030
```
Example output:

`Dense 2D tracks.`




https://github.com/user-attachments/assets/c56d963d-24c0-44f4-beb0-8ba2dee88543


### Step 5/6 Filter Drifting tracks
Since 2D tracks can drift on textureless regions, we discard moving 3D tracks that correspond to certain semantic categories (e.g., `walls`, `building`, `road`, `earth`, `sidewalk`), detected by DeepLabv3 on ADE20K classes.
We can integrate more advanced tracking methods as they become available.

```bash
python segmentation.py \
--vid=CMwZrkhQ0ck_130030030
```
Example output:

`Dense 3D tracks projected onto video frames, without drifting tracks.`


https://github.com/user-attachments/assets/e20d709b-3858-4032-bb06-9d44dde0a8c6





We then fuse these quantities into 4D reconstructions, by lifting the 2D tracks into 3D with their depth.


https://github.com/user-attachments/assets/c52dfe31-7819-4bea-98a8-04c8549d93c1

Since stereo depth estimation is performed per-frame,
the initial disparity estimates (and therefore, the 3D track
positions) are likely to exhibit high-frequency temporal jitter. 

### Step 6/6 Track optimization

To ensure static points remain stationary while moving tracks maintain realistic, smooth motion, 
avoiding abrupt depth changes frame by frame, we design an optimization process (paper Eqn. 5) to get high quality 3D tracks.

```bash
python track_optimization.py \
--vid=CMwZrkhQ0ck_130030030
```
Example output:

`Project the 3D tracks back to get depthmaps.`


https://github.com/user-attachments/assets/198ed277-3658-4ee9-9822-cf55972d6221



`Final 3D tracks` (Color trails are only shown for moving points, but all points have been reconstructed in 3D).



https://github.com/user-attachments/assets/4546b739-058c-4169-ac9a-8a5929885503


🎉 That's it!


Citation
--------
If you find this code useful, please consider citing:

```text
@article{jin2024stereo4d,
  title={Stereo4D: Learning How Things Move in 3D from Internet Stereo Videos}, 
  author={Jin, Linyi and Tucker, Richard and Li, Zhengqi and Fouhey, David and Snavely, Noah and Holynski, Aleksander},
  journal={CVPR},
  year={2025},
}
```

Acknowledgment
--------------
Thanks to Jon Barron, Ruiqi Gao, Kyle Genova, Philipp Henzler, Andrew Liu, Erika Lu, Ben Poole, Qianqian Wang, Rundi Wu, Richard Szeliski, and Stan Szymanowicz for their helpful proofreading, comments, and discussions. Thanks to Carl Doersch, Skanda Koppula, and Ignacio Rocco for their assistance with TAPVid-3D and BootsTAP. Thanks to Carlos Hernandez, Dominik Kaeser, Janne Kontkanen, Ricardo Martin-Brualla, and Changchang Wu for their help with VR180 cameras and videos.
