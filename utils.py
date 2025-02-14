import copy
import numpy as np
import math
import numpy as np
import einops
import tqdm
from typing import List, Optional
import cv2
from matplotlib.collections import LineCollection
import matplotlib

import matplotlib.pyplot as plt

class CameraAZ:
  def __init__(
      self,
      from_json=None,
      from_jaxcam=None,
  ):
    """
    Initialize the object with either JSON data or JAX camera data.

    Parameters:
    from_json (dict, optional): A dictionary containing 'extr' and 'intr_normalized' keys.
    from_jaxcam (jaxcam, optional): Initialize from JAX camera.
    """
    if from_json is not None:
      self.extr = from_json['extr']
      self.intr_normalized = from_json['intr_normalized']
    elif from_jaxcam is not None:
      self._init_from_jaxcam(from_jaxcam)
    else:
      raise NotImplementedError()
    
  def __str__(self):
    return f"extr: \n{self.extr}\n intr_normalized: \n{self.intr_normalized}"

  def _init_from_jaxcam(self, jax_camera):
    self.extr = np.asarray(jax_camera.world_to_camera_matrix[:3])
    self.intr_normalized = {
        'fx': (
            jax_camera.intrinsic_matrix[0][0] / jax_camera.image_size_x
        ).item(),
        'fy': (
            jax_camera.intrinsic_matrix[1][1] / jax_camera.image_size_y
        ).item(),
        'cx': (
            jax_camera.intrinsic_matrix[0][2] / jax_camera.image_size_x
        ).item(),
        'cy': (
            jax_camera.intrinsic_matrix[1][2] / jax_camera.image_size_y
        ).item(),
        'k1': 0,
        'k2': 0,
    }

  def to_json_format(self):
    return {
        'extr': self.extr,
        'intr_normalized': self.intr_normalized,
    }

  def get_c2w(self):
    """
    Get the camera-to-world transformation matrix.
    Returns:
      numpy.ndarray: A 4x4 camera-to-world transformation matrix.
    """
    w2c = np.concatenate((self.extr, np.array([[0, 0, 0, 1]])), axis=0)
    c2w = np.linalg.inv(w2c)
    return c2w

  def get_hfov_deg(self):
    """
    Get the horizontal field of view (HFOV) in degrees.
    """
    return math.degrees(2 * np.arctan(0.5 / self.intr_normalized['fx']))


  def get_intri_matrix(self, imh: int, imw: int):
    """
    Get the intrinsic matrix

    Parameters:
    imh (int): The height of the image.
    imw (int): The width of the image.

    Returns:
    numpy.ndarray: A 3x3 intrinsic matrix.
    """
    return np.array([
        [self.intr_normalized['fx'] * imw, 0, self.intr_normalized['cx'] * imw],
        [0, self.intr_normalized['fy'] * imh, self.intr_normalized['cy'] * imh],
        [0, 0, 1],
    ])
  
  def pix_2_world_np(
      self,
      xy: np.ndarray,
      depth: np.ndarray,
      valid_depth_min: float,
      valid_depth_max: float,
  ):
    """unproject points from ndc from to world frame.

    depth: h x w xy definition:

        xy.shape [:, 2]
        left to right: [0, w]
        top to bottom: [0, h]
    """

    _, dim = xy.shape
    assert dim == 2
    imh, imw = depth.shape

    valid_mask = (
        (xy[:, 0] >= 0) & (xy[:, 1] >= 0) & (xy[:, 0] < imw) & (xy[:, 1] < imh)
    )

    x_cam = (
        xy[..., 0] / imw - self.intr_normalized['cx']
    ) / self.intr_normalized['fx']
    y_cam = (
        xy[..., 1] / imh - self.intr_normalized['cy']
    ) / self.intr_normalized['fy']
    z_cam = np.ones_like(xy[..., 0])
    xyz_cam = np.stack([x_cam, y_cam, z_cam], axis=-1)
    x_query = np.clip(np.round(xy[:, 0]).astype(int), 0, imw - 1)
    y_query = np.clip(np.round(xy[:, 1]).astype(int), 0, imh - 1)
    depth_values = depth[y_query, x_query]

    valid_mask = (
        valid_mask
        & (depth_values > valid_depth_min)
        & (depth_values < valid_depth_max)
    )

    xyz_cam = depth_values[:, None] * xyz_cam

    xyz_world = (self.extr[:3, :3].T @ (xyz_cam - self.extr[:3, 3]).T).T
    return xyz_world, valid_mask

  

  def world_2_pix_np(
      self, xyz_world: np.ndarray, imh: int, imw: int, min_depth: float = 0.01
  ):
    """project points from world frame to screen space.

    xyz_world: [:, 3] array of points in world frame.
    """
    xyz_world_hom = np.concatenate(
        (xyz_world, np.ones_like(xyz_world[:, :1])), axis=-1
    )
    xyz_homo = (
        self.get_intri_matrix(imh, imw) @ self.extr @ xyz_world_hom.T
    ).T  # npt, 3
    depth = xyz_homo[:, 2]
    xy = xyz_homo[:, :2] / xyz_homo[:, 2:]
    valid_mask = (
        (xy[:, 0] >= 0.5)
        & (xy[:, 1] >= 0.5)
        & (xy[:, 0] < imw - 0.5)
        & (xy[:, 1] < imh - 0.5)
        & (depth > min_depth)
    )
    return xy, valid_mask, depth


class Track3d:

  def __init__(
      self,
      tracks: Optional[np.ndarray] = None,
      visibles: Optional[np.ndarray] = None,
      depths: Optional[np.ndarray] = None,
      cameras: Optional[List[CameraAZ]] = None,
      video: Optional[np.ndarray] = None,
      query_points: Optional[np.ndarray] = None,
      valid_depth_min=0,
      valid_depth_max=20,
      load_from_json=None,
      track3d: Optional[np.ndarray] = None,
      visible_list: Optional[np.ndarray] = None,
      color_values: Optional[np.ndarray] = None,
  ):
    """tracks: npt x nframe x d2

    visibles: npt x nframe
    depths: nframe x imh x imw
    cameras: nframe x CameraAZ
    video: nframe x imh x imw
    query_points: npt x 3(t, h, w)
    valid_depth_min: float, min value for valid depth
    valid_depth_max: float, max value for valid depth
    """
    if load_from_json is not None:
      self._load_from_json(load_from_json)
    else:
      # sanity check
      if tracks is not None:
        npt, nframe, d2 = tracks.shape
        assert d2 == 2, 'tracks dimension should be 2'
        assert (npt, nframe) == visibles.shape, f'Wrong shape visibles.shape {visibles.shape}, expected {npt, nframe}' # pytype: disable=attribute-error
        assert depths.shape[0] == nframe, ( # pytype: disable=attribute-error
            f'Wrong shape depths.shape[0] {depths.shape[0]}, expected nframe' # pytype: disable=attribute-error
            f' {nframe}'
        )
        _, imh, imw = depths.shape # pytype: disable=attribute-error
        assert len(cameras) == nframe, f'Wrong shape cameras.shape {len(cameras)}, expected nframe {nframe}'
        assert (nframe, imh, imw, 3) == video.shape, f'Wrong shape video.shape {video.shape}, expected (nframe, imh, imw, 3) {nframe, imh, imw, 3}' # pytype: disable=attribute-error
        if query_points is not None:
          assert query_points.shape == (npt, 3), f'query_points should be npt x 3, got {query_points.shape}'

        # unproject track
        track3d = []
        visible_list = []
        for t in range(len(video)):
          xyz_world, valid_mask = cameras[t].pix_2_world_np(
              tracks[:, t],
              depths[t],
              valid_depth_min,
              valid_depth_max,
          )
          visible_list.append(visibles[:, t] & valid_mask)
          track3d.append(xyz_world)
        track3d = einops.rearrange(
            np.stack(track3d, axis=0), 't npt d3->npt t d3'
        )
        visible_list = einops.rearrange(
            np.stack(visible_list, axis=0), 't npt->npt t'
        )
      elif track3d is not None:
        npt, nframe = track3d.shape[:2]
        assert track3d.shape[2] == 3, f'track3d should be npt x nframe x 3, got {track3d.shape}'
        assert (npt, nframe) == visible_list.shape, f'Wrong shape visible_list.shape {visible_list.shape}, expected {npt, nframe}' # pytype: disable=attribute-error
        assert len(cameras) == nframe, f'Wrong shape cameras.shape {len(cameras)}, expected nframe {nframe}'
        assert nframe == video.shape[0], f'Wrong shape video.shape[0] {video.shape[0]}, expected nframe {nframe}' # pytype: disable=attribute-error
        _, imh, imw, _ = video.shape # pytype: disable=attribute-error
      else:
        raise NotImplementedError
      if color_values is not None:
        pass
      elif query_points is not None:
        # get point color
        color_values = video[
            query_points[:, 0],
            query_points[:, 1].astype(int),
            query_points[:, 2].astype(int),
        ]  # npt, 3
      else:
        color_values = None

      self.cameras = cameras
      self.track3d = track3d
      self.imh = imh
      self.imw = imw
      self.visible_list = visible_list
      self.color_values = color_values
      self.video = video

  def _load_from_json(self, load_from_json):
    if load_from_json['cameras'] is not None:
      self.cameras = [
          CameraAZ(from_json=camera)
          for camera in load_from_json['cameras']  # FIXME
          # for camera in load_from_json['camera']
      ]
    self.track3d = np.stack(load_from_json['track3d'], axis=0)
    self.imh = load_from_json['imh']
    self.imw = load_from_json['imw']
    self.visible_list = load_from_json['visible_list']
    self.color_values = load_from_json['color_values']
    self.video = load_from_json['video']

  def to_json_format(self, save_video=False, save_camera=True):
    return {
        'cameras': [camera.to_json_format() for camera in self.cameras] if save_camera else None,
        'track3d': self.track3d,
        'imh': self.imh,
        'imw': self.imw,
        'visible_list': self.visible_list,
        'color_values': self.color_values,
        'video': self.video if save_video else None,
    }

  def get_new_track(self, track_mask=None, percentage=None | float):
    """
    Generate a new track by applying a mask or a random selection based on a given percentage.

    Parameters:
    track_mask (numpy.ndarray, optional): A boolean mask array to select specific tracks. If None, a random mask will be generated.
    percentage (float, optional): The percentage of tracks to randomly select if track_mask is None. Should be a value between 0 and 1.

    Returns:
    new_track (object): A new instance of the track object with the selected tracks.
    """
    if track_mask is None:
      track_mask = np.random.uniform(size=self.track3d.shape[0]) < percentage
    new_track = copy.deepcopy(self)
    new_track.track3d = new_track.track3d[track_mask]
    new_track.visible_list = new_track.visible_list[track_mask]
    if new_track.color_values is not None:
      new_track.color_values = new_track.color_values[track_mask]
    return new_track
  


def get_scene_motion_2d_displacement(
    track3d: Track3d,
    tracks_leave_trace=16,
):
  """Get 2D point trajectories of scene motion.

  Returns max 2D displacement of 3D points over tracks_leave_trace frames as if
  the camera is static. This measurement decouples camera motion.

  Args:
      track3d: An instance of Track3d containing 3D tracks and visibility info.
      tracks_leave_trace: Number of frames over which to compute displacement.

  Returns:
      displacement: A (npt, nframe) array of max 2D displacements over
      tracks_leave_trace frames.
  """
  all_points = track3d.track3d  # Shape: (npt, nframe, 3)
  npt, nframe, _ = all_points.shape
  displacement = np.zeros_like(track3d.visible_list, dtype=np.float32)
  for t in tqdm.tqdm(range(nframe), desc='Computing 2D displacement'):
    s_start = max(0, t - tracks_leave_trace)
    s_end = t + 1  # Include current frame
    s_list = np.arange(s_start, s_end)  # Shape: (L,)
    L = len(s_list)
    if L < 2:
      # Not enough frames to compute displacement
      continue
    # Extract positions and visibilities for relevant frames
    positions = all_points[:, s_list, :]  # Shape: (npt, L, 3)
    visibilities = track3d.visible_list[:, s_list]  # Shape: (npt, L)
    # Flatten positions for projection
    positions_flat = positions.reshape(-1, 3)
    # Project all positions using the camera at frame t
    points_2d_flat, valid_mask_flat, _ = track3d.cameras[t].world_2_pix_np(
        positions_flat,
        track3d.imh,
        track3d.imw,
    )
    # Reshape back to (npt, L, 2)
    points_2d = points_2d_flat.reshape(npt, L, -1)
    valid_mask = valid_mask_flat.reshape(npt, L)

    # Extract positions and masks at time t
    points_2d_t = points_2d[:, -1, :]  # Shape: (npt, 2)
    valid_mask_t = valid_mask[:, -1]
    visibilities_t = visibilities[:, -1]

    # Compute displacements to previous frames
    deltas = (
        points_2d[:, :-1, :] - points_2d_t[:, None, :]
    )  # Shape: (npt, L-1, 2)
    distances = np.linalg.norm(deltas, axis=2)  # Shape: (npt, L-1)
    # print("distances: ", distances.shape)
    # Validity mask
    valid = (
        valid_mask[:, :-1]
        & valid_mask_t[:, None]
        & visibilities[:, :-1]
        & visibilities_t[:, None]
    )
    # Apply validity mask
    distances[~valid] = 0
    # Compute maximum displacement
    max_displacement = np.max(distances, axis=1)  # Shape: (npt,)
    displacement[:, t] = max_displacement
  return displacement



def flow_to_depth(flow: np.ndarray, hfov_deg, baseline) -> np.ndarray:
  """Calculates depth map from the flow field and camera metadata.

  assumes cx2 - cx1 = 0, valid disparity should be positive

  Args:
      flow: The optical flow field (numpy array).
      hfov_deg: Horizontal field of view (degree).
      baseline: The baseline value in meters (float).

  Returns:
      The calculated depth map (numpy array).
  """
  # disp = np.abs(flow[..., 0])
  disp = np.clip(flow[..., 0], 0, None)
  imh, imw = disp.shape
  fx = imw / np.tan(np.radians(hfov_deg / 2)) / 2
  depth = (fx * baseline) / disp
  return depth



def inverse_warp(img: np.ndarray, flow: np.ndarray) -> np.ndarray:
  """Warps an image based on the provided flow field.

  Args:
      img: The image to warp (numpy array).
      flow: The optical flow field (numpy array).

  Returns:
      The warped image (numpy array).
  """
  h, w = flow.shape[:2]
  flow_new = flow.copy()
  flow_new[:, :, 0] += np.arange(w)
  flow_new[:, :, 1] += np.arange(h)[:, np.newaxis]

  res = cv2.remap(
      img, flow_new, None, cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT
  )
  return res


def plot_3d_tracks_plt(
    video: np.ndarray,
    track3d: Track3d,
    tracks_leave_trace=16,
    point_size: int = 10,
):
  """Visualize 2D point trajectories.

  The trail shows where the previous 3D point in the current camera frame. This
  visualization decouples camera motion
  """
  num_points, num_frames = track3d.track3d.shape[:2]
  figure_dpi = 64

  # Precompute colormap for points
  color_map = matplotlib.colormaps.get_cmap('hsv')
  cmap_norm = matplotlib.colors.Normalize(vmin=0, vmax=num_points - 1)

  # sort points by height of their first appearance
  point_colors = np.zeros((num_points, 3))
  for i in range(num_points):
    point_colors[i] = np.array(color_map(cmap_norm(i)))[:3]

  disp = []
  for t in range(num_frames):
    frame = video[t].copy()

    # Draw tracks on the frame
    all_points = track3d.track3d  # npt, nframe, xyz
    npt, nframe, _ = all_points.shape
    all_points = einops.rearrange(
        all_points, 'npt nframe xyz->(npt nframe) xyz'
    )

    points_at_frame, valid_mask, _ = track3d.cameras[t].world_2_pix_np(
        all_points,
        track3d.imh,
        track3d.imw,
    )
    points_at_frame = einops.rearrange(
        points_at_frame,
        '(npt nframe) xyz->nframe npt xyz',
        npt=npt,
        nframe=nframe,
    )
    valid_mask = einops.rearrange(
        valid_mask, '(npt nframe)-> npt nframe', npt=npt, nframe=nframe
    )
    valid_mask = valid_mask & track3d.visible_list
    valid_mask = valid_mask.transpose(1, 0)
    line_tracks = points_at_frame[max(0, t - tracks_leave_trace) : t + 1]
    line_visibles = valid_mask[max(0, t - tracks_leave_trace) : t + 1]
    fig = plt.figure(
        figsize=(frame.shape[1] / figure_dpi, frame.shape[0] / figure_dpi),
        dpi=figure_dpi,
        frameon=False,
        facecolor='w',
    )
    ax = fig.add_subplot()
    ax.axis('off')
    ax.imshow(frame / 255.0)

    for s in range(line_tracks.shape[0] - 1):
      # Collect lines and colors for the track
      visible_line_mask = line_visibles[s] & line_visibles[s + 1] & line_visibles[-1]
      pt1 = line_tracks[s, visible_line_mask]
      pt2 = line_tracks[s + 1, visible_line_mask]
      lines = np.concatenate([pt1, pt2], axis=1)
      lines = [[(x1, y1), (x2, y2)] for x1, y1, x2, y2 in lines]
      c = point_colors[visible_line_mask]
      alpha = (s + 1) / (line_tracks.shape[0] - 1)
      c = np.concatenate([c, np.ones_like(c[..., :1]) * alpha], axis=1)
      lc = LineCollection(lines, colors=c, linewidths=1)
      ax.add_collection(lc)
    visibles_mask = valid_mask[t].astype(bool)
    colalpha = point_colors[visibles_mask]
    plt.scatter(
        points_at_frame[t, visibles_mask, 0],
        points_at_frame[t, visibles_mask, 1],
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