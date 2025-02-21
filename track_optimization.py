import argparse
import copy
from functools import partial
import jax
import jax.numpy as jnp
import optax
import tqdm
import numpy as np
import math
import os
import os.path as osp
import pickle
from absl import app
from absl import flags
import mediapy as media
import numpy as np
import pickle
import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import matplotlib.pyplot as plt
import json
from typing import List, Optional
import utils


  

def dilate_zeros(mask, window_size):
  """Dilate zeros in a 1D mask array.

  Args:
      mask (jnp.ndarray): 1D array of 0s and 1s.
      window_size (int): Size of the dilation window.

  Returns:
      jnp.ndarray: Dilated mask array.
  """
  # Invert the mask: zeros become ones, ones become zeros
  inv_mask = 1 - mask

  # Calculate padding to keep the output size the same
  pad_before = window_size // 2
  pad_after = window_size - pad_before - 1
  padding = [(pad_before, pad_after)]

  # Apply a maximum filter using reduce_window
  max_filtered = jax.lax.reduce_window(
      inv_mask,
      init_value=0.0,
      computation=jax.lax.max,
      window_dimensions=(window_size,),
      window_strides=(1,),
      padding=padding,
  )

  # Invert back to get the dilated mask
  dilated_mask = 1 - max_filtered

  return dilated_mask


@partial(jax.jit, static_argnums=(1, 2))
def shift_with_edge_padding(array: jnp.ndarray, shift: int, axis: int = 0):
  """Shifts the elements of an array along a specified axis, padding with edge values.

  Parameters:
  - array: The input array to shift.
  - shift: The number of positions to shift. Positive values shift right/down,
  negative shift left/up.
  - axis: The axis along which to shift.

  Returns:
  - A new array with the same shape as the input array, shifted and padded with
  edge values.
  """
  ndim = array.ndim
  axis = axis % ndim  # Ensure axis is within the valid range
  # Compute padding widths
  pad_width = [(0, 0)] * ndim  # Initialize pad widths for all axes
  shift = int(shift)

  pad_before = max(shift, 0)
  pad_after = max(-shift, 0)
  pad_width[axis] = (pad_before, pad_after)

  # Pad the array with edge values
  padded_array = jnp.pad(array, pad_width, mode='edge')

  # Compute start and end indices for slicing
  start = pad_before
  end = start + array.shape[axis]

  # Create slices for all axes
  slicer = [slice(None)] * ndim
  slicer[axis] = slice(start - shift, end - shift)

  # Slice the padded array to get the shifted array
  result = padded_array[tuple(slicer)]
  return result


def compute_static_loss(P_adjusted: jnp.ndarray, masks: jnp.ndarray, w_static: float):
  """
  Compute the static loss for tracks.

  Args:
    P_adjusted (jnp.ndarray): Adjusted points array of shape (N, 3), invalid points are nans.
    masks (jnp.ndarray): Mask array of shape (N,) True indicating valid points.
    w_static (float): Static weight parameter.
  """
  P_adjusted = jnp.nan_to_num(P_adjusted)
  masks = masks.astype(jnp.float32).reshape(-1)  # Shape (N,)
  norm = jnp.nansum(
      jnp.sqrt(jnp.sum(P_adjusted * P_adjusted * masks[:, None], axis=-1))
  ) / (jnp.sum(masks) + 1e-8)
  P_adjusted_masked = (
      P_adjusted / (norm + 1e-5) * masks[:, None]
  )  # Shape (N, 3)
  # Compute pairwise differences
  diffs = (
      P_adjusted_masked[:, None, :] - P_adjusted_masked[None, :, :]
  )  # Shape (N, N, 3)

  # Compute squared distances
  diffs_sqr = jnp.sum(diffs ** 2, axis=-1)  # Shape (N, N)
  # Create pairwise mask: valid if both points are valid
  pairwise_masks = masks[:, None] * masks[None, :]  # Shape (N, N)
  # Apply the pairwise mask to the squared distances
  masked_diffs_sqr = diffs_sqr * pairwise_masks  # Shape (N, N)

  # Compute the number of valid pairs
  num_valid_pairs = jnp.sum(pairwise_masks)  # Scalar
  num_valid_pairs = jnp.maximum(num_valid_pairs, 1.0)

  # Compute the static loss
  static_loss = (w_static / (2 * num_valid_pairs)) * jnp.nansum(
      masked_diffs_sqr
  )
  return static_loss


def compute_dynamic_loss(
    P_adjusted: jnp.ndarray, masks: jnp.ndarray, R_unit: jnp.ndarray, w_dynamic: float, delta: int = 1):
  """
  Computes the dynamic loss for the given adjusted points and masks.

  Parameters:
  P_adjusted (jnp.ndarray): Adjusted points array of shape (N, 3).
  masks (jnp.ndarray): Masks array of shape (N,), True indicating valid points.
  R_unit (jnp.ndarray): Unit vector of camera ray of shape (N, 3).
  w_dynamic (float): Weight for the dynamic loss.
  delta (int): Shift value for edge padding. Default is 1.
  """
  masks = masks.astype(jnp.float32).reshape(-1, 1)
  P_shift_plus = shift_with_edge_padding(
      P_adjusted.transpose(), -delta, axis=1
  ).transpose()
  P_shift_minus = shift_with_edge_padding(
      P_adjusted.transpose(), delta, axis=1
  ).transpose()
  masks_shift_plus = shift_with_edge_padding(
      masks.transpose(), -delta, axis=1
  ).transpose()
  masks_shift_minus = shift_with_edge_padding(
      masks.transpose(), delta, axis=1
  ).transpose()
  valid_mask = masks * masks_shift_plus * masks_shift_minus
  delta2_Pi = P_shift_plus - 2 * P_adjusted + P_shift_minus
  a_i = jnp.nansum(delta2_Pi * R_unit, axis=1, keepdims=True)
  a_i_masked = a_i * valid_mask
  dynamic_loss = jnp.nansum((a_i_masked) ** 2) * w_dynamic
  return dynamic_loss


def compute_regularization_loss(d: jnp.ndarray, ray_distances: np.ndarray, masks: np.ndarray, lambda_reg: float):
  """
  Computes the regularization loss for the given displacement and original ray distances.

  Args:
    d (jnp.ndarray): The displacement.
    ray_distances (np.ndarray): The distances along the rays.
    masks (np.ndarray): The masks indicating valid regions.
    lambda_reg (float): The regularization coefficient.

  Returns:
    float: The computed regularization loss.
  """
  masks = masks.astype(jnp.float32).reshape(-1)
  delta_d = d
  denom = delta_d + ray_distances + 1e-8
  term = ((1 / denom - 1 / (ray_distances + 1e-8)) ** 2) * masks
  reg_loss = lambda_reg * jnp.nansum(term)
  return reg_loss


def loss_fn_inner(
    d: jnp.ndarray,
    P: np.ndarray,
    C: np.ndarray,
    masks: np.ndarray,
    motion_scores: np.ndarray,
    motion_threshold: float,
    w_static: float,
    w_dynamic: float,
    lambda_reg: float,
):
  """
  Computes the total loss for the given parameters.

  Args:
    d (jnp.ndarray): Displacement vector.
    P (np.ndarray): Points in the current frame.
    C (np.ndarray): Points in the previous frame.
    masks (np.ndarray): Binary masks indicating valid points.
    motion_scores (np.ndarray): Motion scores for each point.
    motion_threshold (float): Threshold for motion scores.
    w_static (float): Weight for static loss.
    w_dynamic (float): Weight for dynamic loss.
    lambda_reg (float): Regularization parameter.

  Returns:
    jnp.ndarray: The computed total loss.
  """
  masks_float = masks.astype(jnp.float32).reshape(-1)
  R = P - C
  R_norm = jnp.linalg.norm(R, axis=1, keepdims=True) + 1e-8
  R_unit = R / R_norm
  P_adjusted = P + R_unit * d[:, None]
  ray_distances = jnp.linalg.norm(R, axis=1) + 1e-8
  valid_motion_scores = jnp.where(masks_float > 0.5, motion_scores, jnp.nan)
  motion_percentile = jnp.nanpercentile(valid_motion_scores, 90)
  sigmoid = lambda x: 1 / (1 + jnp.exp(-x))
  w_i = sigmoid(motion_percentile - motion_threshold)
  static_loss = compute_static_loss(P_adjusted, masks, w_static)
  dynamic_loss = (
      compute_dynamic_loss(P_adjusted, masks, R_unit, w_dynamic)
      + compute_dynamic_loss(P_adjusted, masks, R_unit, w_dynamic, delta=3)
      + compute_dynamic_loss(P_adjusted, masks, R_unit, w_dynamic, delta=5)
  )
  reg_loss = compute_regularization_loss(d, ray_distances, masks, lambda_reg)
  total_loss = (1 - w_i) * static_loss + w_i * dynamic_loss + reg_loss
  return total_loss


def optimize_single_track_jax_adam(
    P: np.ndarray,
    C: np.ndarray,
    masks: np.ndarray,
    motion_scores: np.ndarray,
    motion_threshold: float,
    lambda_reg: float,
    w_static: float,
    w_dynamic: float,
    num_steps: int,
    learning_rate: float,
):
  N = P.shape[0]
  d_init = jnp.zeros(N)

  # Define the loss function
  def loss_fn(d, P, C, masks, motion_scores):
    return loss_fn_inner(
        d,
        P,
        C,
        masks,
        motion_scores,
        motion_threshold,
        w_static,
        w_dynamic,
        lambda_reg,
    )

  # Create optimizer
  optimizer = optax.adam(learning_rate)
  opt_state = optimizer.init(d_init)

  # Define a step function
  @jax.jit
  def step(d, opt_state):
    loss, grads = jax.value_and_grad(loss_fn)(d, P, C, masks, motion_scores)
    updates, opt_state = optimizer.update(grads, opt_state)
    d = optax.apply_updates(d, updates)
    return d, opt_state, loss

  # Optimization loop
  d = d_init
  for i in range(num_steps):
    d, opt_state, loss = step(d, opt_state)
  return d



def adjust_positions(P, C, d):
  R = P - C
  R_norm = jnp.linalg.norm(R, axis=-1, keepdims=True) + 1e-8
  R_unit = R / R_norm
  P_adjusted = P + R_unit * d[..., None]
  return P_adjusted


def optimize_tracks(track3d: utils.Track3d):
  """
  Optimize the 3D tracks of points with Adam optimizer.
  Args:
    track3d (Track3D): An object containing 3D tracks, camera information, visibility masks, and color values.
  Returns:
    Track3D: A new Track3D object with optimized 3D tracks, updated visibility masks, and color values.
  """

  points = track3d.track3d  # Batch, N, 3
  cameras = np.array([c.get_c2w()[:3, 3] for c in track3d.cameras])[
      None
  ]  # 1, N, 3
  cameras = np.tile(cameras, (len(points), 1, 1))
  masks = track3d.visible_list[
      ..., None
  ]  # mask of whether the point will be considered in the optimization Batch, N

  dilate_zeros_vmap = jax.vmap(dilate_zeros, in_axes=(0, None))
  window_size = 5
  masks = np.asarray(
      dilate_zeros_vmap(masks[..., 0].astype(jnp.float32), window_size)[
          ..., None
      ].astype(bool)
  )
  colors = track3d.color_values / 255.0
  motion = utils.get_scene_motion_2d_displacement(track3d)
  points = points * masks  # Shape (Batch, N, 3)

  # Optimization hyperparameters.
  w_static = 100.0
  w_dynamic = 100.0
  lambda_reg = 0.1
  motion_threshold = 20
  optimal_batch_size = (
      10000  # Adjust this value according to your memory limits
  )
  num_steps = 100
  learning_rate = 0.05

  num_samples = points.shape[0]
  num_batches = int(np.ceil(num_samples / optimal_batch_size))

  results = []
  optimize_single_track_jax_vmap = jax.vmap(
      optimize_single_track_jax_adam,
      in_axes=(0, 0, 0, 0, None, None, None, None, None, None),
  )
  for i in tqdm.tqdm(range(num_batches), desc='optimizing tracks'):
    start_idx = i * optimal_batch_size
    end_idx = min((i + 1) * optimal_batch_size, num_samples)

    batch_points = points[start_idx:end_idx]
    batch_cameras = cameras[start_idx:end_idx]
    batch_masks = masks[start_idx:end_idx]
    batch_motion = motion[start_idx:end_idx]
    d_batch = optimize_single_track_jax_vmap(
        batch_points,
        batch_cameras,
        batch_masks,
        batch_motion,
        motion_threshold,
        lambda_reg,
        w_static,
        w_dynamic,
        num_steps,
        learning_rate,
    )
    results.append(d_batch)

  # Concatenate results from all batches
  d_batch = jnp.concatenate(results, axis=0)
  # Adjusted positions
  P_adjusted_batch = np.array(adjust_positions(points, cameras, d_batch))

  track3d_new = copy.deepcopy(track3d)
  track3d_new.track3d = np.array(P_adjusted_batch)
  track3d_new.visible_list = masks[..., 0]
  track3d_new.color_values = colors
  return track3d_new


def gradient_check_mask_relative(depth_map, threshold):
  if depth_map.ndim == 2:
    # Case for single depth map with shape (h, w)
    padded_depth_map = np.pad(depth_map, pad_width=1, mode='edge')

    # Compute x and y gradients
    grad_x = np.abs(padded_depth_map[1:-1, 2:] - padded_depth_map[1:-1, :-2])
    grad_y = np.abs(padded_depth_map[2:, 1:-1] - padded_depth_map[:-2, 1:-1])

    # Check if any gradient exceeds the threshold and the pixel is non-zero
    mask = ((grad_x > threshold * depth_map) | (grad_y > threshold * depth_map)) & (depth_map != 0)

  elif depth_map.ndim == 3:
    # Case for batch of depth maps with shape (b, h, w)
    padded_depth_map = np.pad(depth_map, pad_width=((0, 0), (1, 1), (1, 1)), mode='edge')

    # Compute x and y gradients
    grad_x = np.abs(padded_depth_map[:, 1:-1, 2:] - padded_depth_map[:, 1:-1, :-2])
    grad_y = np.abs(padded_depth_map[:, 2:, 1:-1] - padded_depth_map[:, :-2, 1:-1])

    # Check if any gradient exceeds the threshold and the pixel is non-zero
    mask = ((grad_x > threshold * depth_map) | (grad_y > threshold * depth_map)) & (depth_map != 0)

  else:
    raise ValueError("depth_map must have shape (h, w) or (b, h, w)")

  return mask



def load_rgbd_cam_from_pkl(vid: str, root_dir: str, npz_folder: str, hfov: float, new_imw=1, new_imh=1):
  """load rgb, depth, and camera"""
  input_dict = {'left': {'camera': [], 'depth': [], 'video': []}}
  # Load camera
  dp = utils.load_dataset_npz(osp.join(npz_folder, f'{vid}.npz'))
  extrs_rectified = dp['extrs_rectified']

  nfr = len(extrs_rectified)
  input_dict['nfr'] = nfr
  for fid in range(nfr):
    intr_normalized = {
        'fx': (1 / 2.0) / math.tan(math.radians(hfov / 2.0)),
        'fy': (
            (1 / 2.0) / math.tan(math.radians(hfov / 2.0)) * new_imw / new_imh
        ),
        'cx': 0.5,
        'cy': 0.5,
        'k1': 0,
        'k2': 0,
    }
    input_dict['left']['camera'].append(
        utils.CameraAZ(
            from_json={
                'extr': extrs_rectified[fid][:3, :],
                'intr_normalized': intr_normalized,
            }
        )
    )
  video_path = osp.join(
      root_dir,
      vid,
      vid + '-left_rectified.mp4',
  )
  rgbs = media.read_video(video_path)
  input_dict['left']['video'] = rgbs
  flow_path = osp.join(
      root_dir,
      vid,
      # vid + '-left_depth.pkl',
      vid + '-flows_stereo.pkl',
  )
  with open(flow_path, 'rb') as f:
    flows = pickle.load(f)
  depths = []
  for fid in range(nfr):
    flow_fwd = flows[fid]['fwd'].astype(np.float32)
    flow_bwd = flows[fid]['bwd'].astype(np.float32)
    depth = utils.flow_to_depth(
        -flows[fid]['fwd'], input_dict['left']['camera'][fid].get_hfov_deg(), 0.063
    )  # pytype: disable=attribute-error
  
    # Remove occluded points
    flow_bwd_warp = utils.inverse_warp(flow_bwd, flow_fwd)
    occ_mask_left = np.linalg.norm(flow_bwd_warp + flow_fwd, axis=-1) > 1
    depth[occ_mask_left] = 0
    depth[np.abs(flow_fwd[..., 1]) > 1] = 0
    depths.append(depth)
  depths = np.stack(depths, axis=0)
  depths[depths > 20] = 0
  depths[depths < 0] = 0
  # remove floating points
  mask = gradient_check_mask_relative(depths, 0.03)
  depths[mask] = 0
  input_dict['left']['depth'] = depths
  return input_dict


def to_vid(vid: tuple[str, str]):
  return vid[0] + '-clip' + vid[1]

def draw_camera_pose(M, ax, al=1, fov=60, aspect=1.0, near=0.0, far=0.1):
  """Draws the camera pose with axes and frustum in the 3D plot.

  Parameters:
  - M: (4x4 array) The camera's extrinsic matrix.
  - ax: The matplotlib 3D axis to draw on.
  - al: (float) The length scaling factor for the axes.
  - orig_color: (str) The color of the camera origin point.
  - fov: (float) Field of view in degrees.
  - aspect: (float) Aspect ratio of the camera (width / height).
  - near: (float) Near clipping plane distance.
  - far: (float) Far clipping plane distance.
  """
  camera_orig = M[:3, 3]
  rot_mat = M[:3, :3]

  # Draw camera axes
  camera_axes = rot_mat @ np.eye(3)
  ax.quiver(
      camera_orig[0],
      camera_orig[2],
      camera_orig[1],
      camera_axes[0, 0],
      camera_axes[2, 0],
      camera_axes[1, 0],
      length=al,
      color='r',
  )
  ax.quiver(
      camera_orig[0],
      camera_orig[2],
      camera_orig[1],
      camera_axes[0, 1],
      camera_axes[2, 1],
      camera_axes[1, 1],
      length=al,
      color='g',
  )
  ax.quiver(
      camera_orig[0],
      camera_orig[2],
      camera_orig[1],
      camera_axes[0, 2],
      camera_axes[2, 2],
      camera_axes[1, 2],
      length=al,
      color='b',
  )

  # Draw frustum
  fov_rad = np.deg2rad(fov / 2)
  h_near = 2 * np.tan(fov_rad) * near
  w_near = h_near * aspect
  h_far = 2 * np.tan(fov_rad) * far
  w_far = h_far * aspect

  # Define frustum corners in camera space
  near_corners = np.array([
      [w_near / 2, h_near / 2, near],
      [-w_near / 2, h_near / 2, near],
      [-w_near / 2, -h_near / 2, near],
      [w_near / 2, -h_near / 2, near],
  ])

  far_corners = np.array([
      [w_far / 2, h_far / 2, far],
      [-w_far / 2, h_far / 2, far],
      [-w_far / 2, -h_far / 2, far],
      [w_far / 2, -h_far / 2, far],
  ])

  # Combine near and far corners
  frustum_corners = np.vstack((near_corners, far_corners))

  # Transform corners to world space
  frustum_corners_world = (rot_mat @ frustum_corners.T).T + camera_orig

  # Define lines to draw the frustum edges
  frustum_lines = (
      [
          # Lines from camera origin to near corners
          [camera_orig, frustum_corners_world[i]]
          for i in range(4)
      ]
      + [
          # Lines from camera origin to far corners
          [camera_orig, frustum_corners_world[i + 4]]
          for i in range(4)
      ]
      + [
          # Edges of the near plane
          [frustum_corners_world[i], frustum_corners_world[(i + 1) % 4]]
          for i in range(4)
      ]
      + [
          # Edges of the far plane
          [
              frustum_corners_world[i + 4],
              frustum_corners_world[((i + 1) % 4) + 4],
          ]
          for i in range(4)
      ]
      + [
          # Edges connecting near and far planes
          [frustum_corners_world[i], frustum_corners_world[i + 4]]
          for i in range(4)
      ]
  )

  # Convert to numpy array and adjust axes for plotting
  frustum_lines = np.array(frustum_lines)
  frustum_lines = frustum_lines[:, :, [0, 2, 1]]  # Swap y and z axes


  frustum_collection = Line3DCollection(
      frustum_lines, colors='gray', linewidths=1
  )
  ax.add_collection3d(frustum_collection)




def save_3d_track_vis(
    track3d: utils.Track3d,
    rgbs: np.ndarray,
    depth: np.ndarray, 
    save_root: str,
    vid: str,
    prefix: str,
):
  motion_mag = utils.get_scene_motion_2d_displacement(track3d)
  track3d_dynamic = track3d.get_new_track((motion_mag > 16).any(axis=1))
  location = track3d_dynamic.track3d[
      np.arange(len(track3d_dynamic.track3d)),
      np.argmax((~np.isnan(track3d_dynamic.track3d).any(axis=-1) & track3d_dynamic.visible_list), axis=1),
  ]
  index = np.argsort(location[:, 1])
  track3d_dynamic_sort = track3d_dynamic.get_new_track(index)

  # pick tracks that has longest visibilities
  visible_length = track3d_dynamic_sort.visible_list.sum(axis=1)
  masks = np.zeros(len(track3d_dynamic_sort.track3d))
  masks[visible_length > np.percentile(visible_length, 10)] = 1
  masks = masks.astype(bool)
  track3d_dynamic_sort_sampled = track3d_dynamic_sort.get_new_track(masks)
  track3d_dynamic_sort_sampled = track3d_dynamic_sort.get_new_track(
      percentage=min(256 / len(track3d_dynamic_sort_sampled.track3d), 1)
  )

  video3d_viz = plot_3d_tracks_with_camera(
    track3d_dynamic_sort_sampled,
    depth,
    rgbs,
    axes=None,
    tracks_leave_trace=16
  )

  dyna_track_plt = utils.plot_3d_tracks_plt(
      rgbs,
      track3d_dynamic_sort_sampled,
  )
  media.write_video(
      osp.join(save_root, vid, vid + f'-dyna_3dtrack-{prefix}.mp4'), video3d_viz, fps=30
  )
  media.write_video(
      osp.join(save_root, vid, vid + f'-dyna_2dtrack-{prefix}.mp4'),
      dyna_track_plt,
      fps=30,
  )
  media.write_video(
      osp.join(save_root, vid, vid + f'-dyna_3dtrack_concated-{prefix}.mp4'),
      np.concatenate([dyna_track_plt / 255.0, video3d_viz], axis=2),
      fps=30,
  )
  return osp.join(save_root, vid, vid + '-dyna_3dtrack_concated.mp4')

def colored_depthmap(
    depth: np.ndarray,
    d_min: Optional[float] = None,
    d_max: Optional[float] = None,
    invalid_value: Optional[float] = None,
    colormap='Spectral',
) -> np.ndarray:
  """Converts a depth map to a colored image using a plasma colormap.

  Args:
    depth: The depth map (numpy array).
    d_min: Minimum depth value (float, optional). Defaults to None (minimum in
      depth).
    d_max: Maximum depth value (float, optional). Defaults to None (maximum in
      depth).

  Returns:
    The colored depth map as a numpy array of uint8 representing RGB channels.
  """
  if d_min is None:
    d_min = np.min(depth)
  if d_max is None:
    d_max = np.max(depth)
  depth_relative = (depth - d_min) / (d_max - d_min)
  cmap = getattr(plt.cm, colormap, None)
  if cmap is not None:
    depth_colored = 255 * cmap(depth_relative)[:, :, :3]  # H, W, C
  else:
    raise ValueError(f"Colormap '{colormap}' is not recognized")
  depth_colored = depth_colored.astype(np.uint8)
  depth_colored[depth == invalid_value] = 0
  return depth_colored

def plot_depth_prism(depthmaps):
  depthmaps[depthmaps == 0] = np.nan

  vmin = np.nanpercentile(depthmaps, 5)
  vmax = np.nanpercentile(depthmaps, 95)
  colored_d = np.stack([colored_depthmap(1/d, 1/vmax, 1/vmin, np.nan, 'turbo') for d in depthmaps], axis=0)
  return colored_d

def draw_camera_pose(M, ax, al=1, fov=60, aspect=1.0, near=0.0, far=0.1):
  """
  Draws the camera pose with axes and frustum in the 3D plot.

  Parameters:
  - M: (4x4 array) The camera's extrinsic matrix.
  - ax: The matplotlib 3D axis to draw on.
  - al: (float) The length scaling factor for the axes.
  - orig_color: (str) The color of the camera origin point.
  - fov: (float) Field of view in degrees.
  - aspect: (float) Aspect ratio of the camera (width / height).
  - near: (float) Near clipping plane distance.
  - far: (float) Far clipping plane distance.
  """
  camera_orig = M[:3, 3]
  rot_mat = M[:3, :3]
  # Draw frustum
  fov_rad = np.deg2rad(fov / 2)
  h_near = 2 * np.tan(fov_rad) * near
  w_near = h_near * aspect
  h_far = 2 * np.tan(fov_rad) * far
  w_far = h_far * aspect

  # Define frustum corners in camera space
  near_corners = np.array([
      [ w_near / 2,  h_near / 2, near],
      [-w_near / 2,  h_near / 2, near],
      [-w_near / 2, -h_near / 2, near],
      [ w_near / 2, -h_near / 2, near],
  ])

  far_corners = np.array([
      [ w_far / 2,  h_far / 2, far],
      [-w_far / 2,  h_far / 2, far],
      [-w_far / 2, -h_far / 2, far],
      [ w_far / 2, -h_far / 2, far],
  ])

  # Combine near and far corners
  frustum_corners = np.vstack((near_corners, far_corners))

  # Transform corners to world space
  frustum_corners_world = (rot_mat @ frustum_corners.T).T + camera_orig

  # Define lines to draw the frustum edges
  frustum_lines = [
      # Lines from camera origin to near corners
      [camera_orig, frustum_corners_world[i]] for i in range(4)
  ] + [
      # Lines from camera origin to far corners
      [camera_orig, frustum_corners_world[i + 4]] for i in range(4)
  ] + [
      # Edges of the near plane
      [frustum_corners_world[i], frustum_corners_world[(i + 1) % 4]] for i in range(4)
  ] + [
      # Edges of the far plane
      [frustum_corners_world[i + 4], frustum_corners_world[((i + 1) % 4) + 4]] for i in range(4)
  ] + [
      # Edges connecting near and far planes
      [frustum_corners_world[i], frustum_corners_world[i + 4]] for i in range(4)
  ]

  # Convert to numpy array and adjust axes for plotting
  frustum_lines = np.array(frustum_lines)
  frustum_lines = frustum_lines[:, :, [0, 2, 1]]  # Swap y and z axes

  # Plot frustum using Line3DCollection
  frustum_collection = Line3DCollection(frustum_lines, colors=[0.0, 0.0, 0.7], linewidths=1)
  ax.add_collection3d(frustum_collection)


def world_points(camera, depth, rgb, stride=1):
    (height, width) = depth.shape
    # Pixel centers
    x = np.arange(0, width, stride) + 0.5
    y = np.arange(0, height, stride) + 0.5
    xv, yv = np.meshgrid(x, y)
    xy = np.stack([xv, yv], axis=-1)
    points, valid = camera.pix_2_world_np(
      xy.reshape(-1, 2),
      depth,
      valid_depth_min=0,
      valid_depth_max=np.inf
    )
    colors = np.reshape(rgb[::stride, ::stride], [-1, 3]) / 255.0
    alphas = np.ones_like(colors[..., 0:1])
    return points[..., [0,2,1]], valid, np.concat((colors, alphas), axis=1)


def plot_3d_tracks_with_camera(track3d: utils.Track3d, depth, rgb, axes=None, tracks_leave_trace=32):
  """Visualize 3D point trajectories with camera trajectory."""
  points = track3d.track3d.transpose(1,0,2)
  visibles = track3d.visible_list.transpose(1,0)
  M_list = [c.get_c2w()[:3] for c in track3d.cameras]

  num_frames, num_points = points.shape[0:2]
  points = points[..., [0,2,1]]

  # Colormap for points
  point_color_map = matplotlib.colormaps.get_cmap('hsv')
  point_cmap_norm = matplotlib.colors.Normalize(vmin=0, vmax=num_points - 1)

  # Colormap for camera trajectory
  camera_color_map = matplotlib.colormaps.get_cmap('Greys')

  if axes and 'x' in axes:
    x_min, x_max = axes['x']
  else:
    x_min, x_max = np.nanpercentile(points[visibles, 0], 5), np.nanpercentile(points[visibles, 0], 95)

  if axes and 'y' in axes:
    y_min, y_max = axes['y']
  else:
    y_min, y_max = np.nanpercentile(points[visibles, 1], 5), np.nanpercentile(points[visibles, 1], 95)

  if axes and 'z' in axes:
    z_min, z_max = axes['z']
  else:
    z_min, z_max = np.nanpercentile(points[visibles, 2], 5), np.nanpercentile(points[visibles, 2], 95)

  # Adjust axis limits
  interval = np.max([x_max - x_min, y_max - y_min, z_max - z_min]) * 1.11
  # print(interval)
  x_mid = (x_min + x_max) / 2
  y_mid = (y_min + y_max) / 2
  z_mid = (z_min + z_max) / 2
  x_min, x_max = x_mid - interval / 2, x_mid + interval / 2
  y_min, y_max = y_mid - interval / 2, y_mid + interval / 2
  z_min, z_max = z_mid - interval / 2, z_mid + interval / 2

  # print(f'Axes: [{x_min}, {x_max}], [{y_min}, {y_max}], [{z_min}, {z_max}]')

  # Precompute camera positions and segments
  if M_list is not None:
    camera_positions = np.array([M[:3, 3] for M in M_list])  # Shape: (num_frames, 3)
    camera_positions = camera_positions[:, [0, 2, 1]]
    camera_segments = np.stack([camera_positions[:-1], camera_positions[1:]], axis=1)  # Shape: (num_frames - 1, 2, 3)

  frames = []
  for t in tqdm.tqdm(range(num_frames), desc='plotting tracks'):
    fig = Figure(figsize=(5.12, 5.12), dpi=100)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111, projection='3d', computed_zorder=False)

    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.set_zlim([z_min, z_max])

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    ax.invert_zaxis()
    ax.view_init()

    # Plot background points
    background_points, valid, colors = world_points(
      track3d.cameras[t], depth[t], rgb[t], stride=2)
    background = ax.scatter(
      background_points[..., 0],
      background_points[..., 1],
      background_points[..., 2],
      c=colors, s=16, depthshade=False, edgecolors='none')

    # Plot camera trajectory up to frame t
    if M_list is not None:
      if False and t > 0:
        segments_to_plot = camera_segments[:t]
        camera_cmap_norm = matplotlib.colors.Normalize(vmin=-1, vmax=t)
        camera_colors = camera_color_map(camera_cmap_norm(np.arange(t-1)))
        colors_to_plot = camera_colors[:t]
        camera_trajectory = Line3DCollection(segments_to_plot, colors=colors_to_plot, linewidths=1)
        ax.add_collection3d(camera_trajectory)

      path_pos = camera_positions[0:np.min((t+20, num_frames))]
      ax.plot(path_pos[..., 0], path_pos[..., 1], path_pos[..., 2],
              color=[0.0, 0.0, 0.7],
              linestyle='dashed')
      # Plot current camera pose
      draw_camera_pose(M_list[t], ax, al=interval/10, far=interval/10)

    # Compute mask for visible points
    mask = visibles[t, :]
    indices = np.where(mask)[0]

    if indices.size > 0:
      start_t = max(0, t - tracks_leave_trace)

      # Extract lines in a vectorized manner
      lines = points[start_t:t+1, indices]  # Shape: (line_length, num_visible_points, 3)
      lines = np.transpose(lines, (1, 0, 2))  # Shape: (num_visible_points, line_length, 3)

      # Prepare colors
      colors = point_color_map(point_cmap_norm(indices))

      # Plot lines using Line3DCollection
      line_collection = Line3DCollection(lines, colors=colors, linewidths=1)
      ax.add_collection3d(line_collection)

      # Plot scatter points
      end_points = points[t, indices]
      ax.scatter(end_points[:, 0], end_points[:, 1], end_points[:, 2], c=colors, s=3)

    fig.subplots_adjust(left=-0.05, right=1.05, top=1.05, bottom=-0.05)
    # Draw including background
    fig.canvas.draw()
    with_background = np.array(canvas.buffer_rgba(), dtype=np.float32) / 255.
    background.set_visible(False)
    fig.canvas.draw()
    without_background = np.array(canvas.buffer_rgba(), dtype=np.float32) / 255.
    k = .4
    blend = with_background * k + without_background * (1-k)
    frames.append(blend)
    plt.close(fig)  # Close the figure to free memory

  return np.array(frames)[..., :3]



def optimize_track_main(vid: str, save_root: str, npz_folder: str, hfov: float):
  # load raw tracks
  with open(
      osp.join(save_root, vid, vid + '-tapir_remove_drift_tracks.pkl'),
      'rb',
  ) as f:
    track2d = pickle.load(f)
  # load depth
  input_dict = load_rgbd_cam_from_pkl(vid, save_root, npz_folder, hfov)
  media.write_video(
    osp.join(save_root, vid, vid + '-raw_depth.mp4'),
    plot_depth_prism(input_dict['left']['depth']),
    fps=30,
  )

  track3d = utils.Track3d(
      track2d['tracks'],
      track2d['visibles'],
      input_dict['left']['depth'],
      input_dict['left']['camera'],
      input_dict['left']['video'],
      track2d['query_points'],
  )
  save_3d_track_vis(track3d, input_dict['left']['video'], input_dict['left']['depth'], save_root, vid, 'original')

  print(f'optimization start for {vid}')
  track3d_new = optimize_tracks(track3d)
  track3d_json = track3d_new.to_json_format(save_video=False)

  print(f'optimization finished for {vid}')
  if not osp.exists(osp.join(save_root, vid)):
    print(f'create folder {osp.join(save_root, vid)}')
    os.makedirs(osp.join(save_root, vid))
  print(
      f"writing to {osp.join(save_root, vid, vid + '-optimized_tracks.pkl')}"
  )
  with open(
      osp.join(save_root, vid, vid + '-optimized_tracks.pkl'),
      'wb',
  ) as f:
    pickle.dump(track3d_json, f)

  # visualize optimized depth from track
  depth_optimized = np.zeros_like(input_dict['left']['depth'])
  for fid in range(len(input_dict['left']['video'])):
    points_at_frame, valid_mask, depth = input_dict['left']['camera'][fid].world_2_pix_np(
      track3d_new.track3d[:, fid],
      track3d_new.imh,
      track3d_new.imw,
    )
    valid_mask = valid_mask & track3d_new.visible_list[:, fid]
    points_at_frame = points_at_frame[valid_mask]
    depth = depth[valid_mask]
    depth_optimized[fid, points_at_frame[:, 1].astype(int), points_at_frame[:, 0].astype(int)] = depth

  media.write_video(
    osp.join(save_root, vid, vid + '-optimized_depth.mp4'),
    plot_depth_prism(depth_optimized), 
    fps=30,
  )

  save_3d_track_vis(track3d_new, input_dict['left']['video'], depth_optimized, save_root, vid, 'optimized')

  


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--vid', help='video id, in the format of <raw-video-id>_<timestamp>', type=str)
  parser.add_argument('--npz_folder', help='npz folder', type=str, default='stereo4d_dataset/npz')
  parser.add_argument('--output_folder', help='output folder', type=str, default='stereo4d_dataset/processed')
  parser.add_argument('--hfov', help='camera field of view', type=float, default=60)

  args = parser.parse_args()

  optimize_track_main(args.vid, args.output_folder, args.npz_folder, args.hfov)

if __name__ == '__main__':
  main()