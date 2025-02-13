import jax
import matplotlib
import matplotlib.pyplot as plt
import mediapy as media
import numpy as np
import sys
sys.path.append('./tapnet')
from tapnet.models import tapir_model
from tapnet.utils import model_utils
from tapnet.utils import transforms
import os
import os.path as osp
import pickle
import tqdm
from scipy.spatial import KDTree
from matplotlib.collections import LineCollection

import argparse
import requests


class TapirTracker:
  """
  Tapir wrapper
  """
  def __init__(self):
    checkpoint_path = 'tapnet/tapnet/checkpoints/bootstapir_checkpoint_v2.npy'
    if not osp.exists(checkpoint_path):
      os.makedirs(osp.dirname(checkpoint_path), exist_ok=True)
      # download the checkpoint
      checkpoint_url = "https://storage.googleapis.com/dm-tapnet/bootstap/bootstapir_checkpoint_v2.npy"
      print(f"Checkpoint not found at {checkpoint_path}. Downloading...")
      try:
        response = requests.get(checkpoint_url, stream=True)
        response.raise_for_status()
        # Save the file
        with open(checkpoint_path, 'wb') as f:
          for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
        print(f"Checkpoint downloaded and saved to {checkpoint_path}")
      except requests.exceptions.RequestException as e:
        print(f"Failed to download checkpoint: {e}")
    ckpt_state = np.load(checkpoint_path, allow_pickle=True).item()
    params, state = ckpt_state['params'], ckpt_state['state']

    kwargs = dict(bilinear_interp_with_depthwise_conv=False, pyramid_level=0)
    kwargs.update(
        dict(pyramid_level=1, extra_convs=True, softmax_temperature=10.0)
    )
    self.model = tapir_model.ParameterizedTAPIR(params, state, tapir_kwargs=kwargs)
    self.chunk_size = 32
    self.resize_height = 512
    self.resize_width = 512
    

  def inference(self, video: np.ndarray, query_points: np.ndarray):
    """Args:

      video: t x h x w x 3
      query_points: n_pt, 3(t x h x w)

    Returns: tracks and visiblity
    track is in npt, nframe, 2(x, y)
    """
    height, width = video.shape[1:3]
    query_points = transforms.convert_grid_coordinates(
        query_points,
        (1, height, width),
        (1, self.resize_width, self.resize_height),
        coordinate_format="tyx",
    )
    frames = media.resize_video(video, (self.resize_height, self.resize_width))
    frames = model_utils.preprocess_frames(frames[None])
    feature_grids = self.model.get_feature_grids(frames, is_training=False)  # pytype: disable=attribute-error

    def chunk_inference(query_points):
      query_points = query_points.astype(np.float32)[None]

      outputs = self.model(
          video=frames,
          query_points=query_points,
          is_training=False,
          query_chunk_size=self.chunk_size,
          feature_grids=feature_grids,
      )
      tracks, occlusions, expected_dist = (
          outputs["tracks"],
          outputs["occlusion"],
          outputs["expected_dist"],
      )

      # Binarize occlusions
      visibles = model_utils.postprocess_occlusions(occlusions, expected_dist)
      return tracks[0], visibles[0]

    chunk_inference = jax.jit(chunk_inference)

    all_tracks = []
    all_visibles = []
    for chunk in tqdm.tqdm(
        range(0, query_points.shape[0], self.chunk_size), desc='chunk inference'
    ):
      tracks, visibles = chunk_inference(
          query_points[chunk : chunk + self.chunk_size]
      )
      all_tracks.append(np.array(tracks))
      all_visibles.append(np.array(visibles))

    tracks = np.concatenate(all_tracks, axis=0)
    visibles = np.concatenate(all_visibles, axis=0)

    # Visualize sparse point tracks
    tracks = transforms.convert_grid_coordinates(
        tracks, (self.resize_width, self.resize_height), (width, height)
    )
    rst = {
        "visibles": visibles,
        "tracks": tracks,
    }
    return rst


def sample_grid_points(
    frame_idx: int,
    height: int,
    width: int,
    stride: int = 1,
    jitter_on=False,
):
  """Sample grid points with (time height, width) order."""
  points = np.mgrid[stride // 2 : height : stride, stride // 2 : width : stride]
  if stride > 1 and jitter_on:
    points = points + np.random.uniform(-stride // 2, stride // 2, points.shape)
    points[0] = np.clip(points[0], 0, height - 1)
    points[1] = np.clip(points[1], 0, width - 1)
  points = points.transpose(1, 2, 0)
  out_height, out_width = points.shape[0:2]
  frame_idx = np.ones((out_height, out_width, 1)) * frame_idx
  points = np.concatenate((frame_idx, points), axis=-1).astype(np.int32)
  points = points.reshape(-1, 3)  # [out_height*out_width, 3]
  return points


def get_min_distance(query_points, other_points):
  tree = KDTree(other_points)
  distances, _ = tree.query(query_points, k=1)
  return distances


def remove_redundant_query(track, query_points, threshold):
  """input:

  track['tracks']: [npt, nframe, xy]
  track['visibles']: [npt, nframe]
  query_points: [npt, tyx]
  threshold: float distance in pixel space
  output:
  track['tracks']: [npt, nframe, xy]
  track['visibles']: [npt, nframe]
  """
  tracks = track['tracks']
  visibles = track['visibles']
  npt, nframe = visibles.shape
  good_points = np.zeros(npt, dtype=bool)
  for fid in np.unique(query_points[:, 0]):
    query_at_fid = query_points[query_points[:, 0] == fid][:, 1:][..., ::-1]
    query_pt_id = np.arange(npt)[query_points[:, 0] == fid]
    tracked_at_fid = tracks[good_points]
    tracked_at_fid_visible = visibles[good_points]

    if len(tracked_at_fid) == 0:
      good_points[query_pt_id] = True
      continue
    tracked_at_fid = tracked_at_fid[:, fid]
    tracked_at_fid_visible = tracked_at_fid_visible[:, fid]
    tracked_at_fid = tracked_at_fid[tracked_at_fid_visible]
    if len(tracked_at_fid) == 0:
      good_points[query_pt_id] = True
      continue
    min_distance = get_min_distance(query_at_fid, tracked_at_fid)
    good_points[query_pt_id[min_distance > threshold]] = True
    
  print(
      f'removed {(1 - good_points.sum() / len(good_points)) * 100 :.2f}%'
      ' redundant points'
  )
  query_points = query_points[good_points]
  new_track = {
      'tracks': tracks[good_points],
      'visibles': visibles[good_points],
  }
  return new_track, query_points


def plot_2d_tracks_plt(
    video: np.ndarray,
    points: np.ndarray,
    visibles: np.ndarray,
    tracks_leave_trace: int = 16,
    point_size: int = 10,
):
  """
  Visualize 2D point trajectories.
  video: nframe x h x w x 3
  points: npt x nframe x 2(xy)
  visibles: npt x nframe, a mask which indicates which point is visible
  """
  points = points.transpose(1, 0, 2)
  visibles = visibles.transpose(1, 0)
  num_frames, num_points = points.shape[:2]
  
  figure_dpi = 64
  # Precompute colormap for points
  color_map = matplotlib.colormaps.get_cmap('hsv')
  cmap_norm = matplotlib.colors.Normalize(vmin=0, vmax=num_points - 1)
  point_colors = np.zeros((num_points, 3))
  for i in range(num_points):
    point_colors[i] = np.array(color_map(cmap_norm(i)))[:3]

  disp = []
  for t in range(num_frames):
    frame = video[t].copy()

    fig = plt.figure(
        figsize=(frame.shape[1] / figure_dpi, frame.shape[0] / figure_dpi),
        dpi=figure_dpi,
        frameon=False,
        facecolor='w',
    )
    ax = fig.add_subplot()
    ax.axis('off')
    ax.imshow(frame / 255.0)

    line_tracks = points[max(0, t - tracks_leave_trace) : t + 1]
    line_visibles = visibles[max(0, t - tracks_leave_trace) : t + 1].astype(
        bool
    )
    for s in range(line_tracks.shape[0] - 1):
      # Collect lines and colors for the track
      visible_line_mask = line_visibles[s] & line_visibles[s + 1]
      pt1 = line_tracks[s, visible_line_mask]
      pt2 = line_tracks[s + 1, visible_line_mask]
      lines = np.concatenate([pt1, pt2], axis=1)
      lines = [[(x1, y1), (x2, y2)] for x1, y1, x2, y2 in lines]
      c = point_colors[visible_line_mask]
      alpha = (s + 1) / (line_tracks.shape[0] - 1)
      c = np.concatenate([c, np.ones_like(c[..., :1]) * alpha], axis=1)
      lc = LineCollection(lines, colors=c, linewidths=1)
      ax.add_collection(lc)
    visibles_mask = visibles[t].astype(bool)
    colalpha = point_colors[visibles_mask]
    plt.scatter(
        points[t, visibles_mask, 0],
        points[t, visibles_mask, 1],
        s=point_size,
        c=colalpha,
    )

    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    fig.canvas.draw()
    img = np.array(fig.canvas.renderer.buffer_rgba())[..., :3]  # pytype: disable=attribute-error
    disp.append(np.copy(img))
    plt.close(fig)
    del fig, ax

  disp = np.stack(disp, axis=0)

  return disp


def run_track(video_id: str, clip_id: int, save_root: str):
  """
  Run 2D track on a video clip and save the results.

  The function performs the following steps:
  1. Reads the perspective video.
  2. Initializes the TapirTracker.
  3. Samples query points.
  4. Runs the TapirTracker on the video and query points.
  5. Removes redundant query points from the tracking results.
  6. Samples a subset of the tracks for visualization.
  7. Writes the resulting video with plotted tracks to the specified path.
  8. Saves the tracking results and query points to a pickle file.
  """
  # Read the video.
  vid = f'{video_id}-clip{clip_id}'
  video = media.read_video(osp.join(save_root, vid, f"{vid}-left_rectified.mp4"))
  # Initialize the TapirTracker
  tapir = TapirTracker()
  tapir.resize_height = 512
  tapir.resize_width = 512
  tapir.chunk_size = 200
  grid_size = 4
  height, width = video.shape[1], video.shape[2]
  # Samples query points
  print('Get query')
  # For every 10th frame, uniformly initialize 128 x 128 query points on frames of resolution 512 x 512. 
  query_points = []
  for frame_id in range(0, video.shape[0], 10):
    query_points_tmp = sample_grid_points(
        frame_id, height, width, grid_size, jitter_on=True
    )
    query_points.append(query_points_tmp)
  query_points = np.concatenate(query_points, axis=0)
  # Run the TapirTracker
  print('raw tapir inference')
  track2d = tapir.inference(video, query_points)
  
  # Prune redundant tracks that overlap on the same pixel.
  print('remove_redundant_query')
  track2d, query_points = remove_redundant_query(
      track2d, query_points, threshold=1
  )
  # Sample a subset of the tracks for visualization
  sampled_points = np.random.choice(
      np.arange(len(track2d['tracks'])), 64 * 64, replace=False
  )
  video_plot = plot_2d_tracks_plt(
      video,
      track2d['tracks'][sampled_points],
      track2d['visibles'][sampled_points],
  )
  media.write_video(
      osp.join(
          save_root,
          f'{video_id}-clip{clip_id}',
          f'{video_id}-clip{clip_id}-tapir_2d.mp4',
      ),
      video_plot,
      fps=30,
  )
  track2d['query_points'] = query_points
  # Save the tracking results and query points to a pickle file
  with open(
      osp.join(
          save_root,
          f'{video_id}-clip{clip_id}',
          f'{video_id}-clip{clip_id}-tapir_2d.pkl',
      ),
      'wb',
  ) as f:
    pickle.dump(track2d, f)



def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--videoid', help='video id', type=str, default='')
  parser.add_argument('--clipid', help='clip id', type=int, default=0)
  parser.add_argument('--output_folder', help='output folder', type=str, default='')

  args = parser.parse_args()

  run_track(args.videoid, args.clipid, args.output_folder)

if __name__ == '__main__':
  main()