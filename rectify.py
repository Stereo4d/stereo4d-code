import argparse
import json
import math
import os
import os.path as osp
from moviepy.video.io.VideoFileClip import VideoFileClip

from absl import app
from absl import flags
from absl import logging
import cv2
import jax
from jax import numpy as jnp
import jaxcam
import mediapy as media
import numpy as np
import tqdm
from tqdm.contrib.concurrent import process_map

jax.config.update('jax_platform_name', 'cpu')

class CustomExceptionName(Exception):
  """Still an exception raised when uncommon things happen"""

  def __init__(self, message, payload=None):
    self.message = message
    self.payload = payload  # you could add more args

  def __str__(self):
    return str(
        self.message
    )  # __str__() obviously expects a string to be returned, so make sure not to send any other data types

class EquiVideoLoader:
  def __init__(self, video_id, raw_video_folder):
    self.video_path = osp.join(raw_video_folder, video_id + '.mp4')

  def retrieve_frame_cv2(self, timestamps):
    vidcap = cv2.VideoCapture(self.video_path)
    print(self.video_path)
    video = []
    for timestamp in tqdm.tqdm(timestamps, desc='Extract frames'):
      vidcap.set(cv2.CAP_PROP_POS_MSEC, timestamp / 1000)
      success, image = vidcap.read()
      # assert success
      if not success:
        raise CustomExceptionName('vidcap error', 'vidcap return not success')
      video.append(image[..., ::-1])
    return np.stack(video, axis=0)

  def retrieve_frames_moviepy(self, timestamps):
    print(self.video_path)
    video = []
    with VideoFileClip(self.video_path) as clip:
      for timestamp in tqdm.tqdm(timestamps, desc='Extract frames'):
        try:
          frame = clip.get_frame(timestamp / 1000000)  # Convert to seconds
          video.append(frame)
        except Exception as e:
          raise CustomExceptionName('frame retrieval error', str(e))
    return np.stack(video, axis=0)



def get_equirect_rectification_map(
    equirect_hw: tuple[int, int], meta_fov: dict, stereo2rig: np.ndarray
):
  """Generate a UV map from rectified equirectangular image to raw equirectangular image.

  Parameters:
  -----------
  equirect_hw : tuple[int, int]
      The height and width of the equirectangular image as a tuple (height,
      width).

  meta_fov : dict
      A dictionary containing the field of view (FOV) parameters in degrees. The
      dictionary should have the following keys:
      - 'start_yaw_in_degrees' : float
          The starting yaw angle in degrees.
      - 'end_yaw_in_degrees' : float
          The ending yaw angle in degrees.
      - 'start_tilt_in_degrees' : float
          The starting tilt angle (pitch) in degrees.
      - 'end_tilt_in_degrees' : float
          The ending tilt angle (pitch) in degrees.

  stereo2rig : np.ndarray
      Calibration matrix from stereo frame to rig frame.

  Returns:
  --------
  map_x : np.ndarray
      A 2D array of x-coordinates in the equirectangular image corresponding to
      the camera's perspective.

  map_y : np.ndarray
      A 2D array of y-coordinates in the equirectangular image corresponding to
      the camera's perspective.

  Notes:
  ------
  - The function creates a UV map that can be used to transform an
  equirectangular image into a rectified equirectangular view based on the
  stereo2rig calibration matrix.
  - The function uses numpy for calculations and meshgrid creation.
  - The resulting UV map (map_x, map_y) can be used for sampling pixels from the
  equirectangular image to create the perspective image.
  """
  # Create coordinate grid
  longitude_stereo = np.linspace(
      np.radians(meta_fov['start_yaw_in_degrees']),
      np.radians(meta_fov['end_yaw_in_degrees']),
      equirect_hw[1],
  )
  latitude_stereo = np.linspace(
      np.radians(meta_fov['start_tilt_in_degrees']),
      np.radians(meta_fov['end_tilt_in_degrees']),
      equirect_hw[0],
  )
  xv, yv = np.meshgrid(longitude_stereo, latitude_stereo)
  ray_x = np.cos(yv) * np.sin(xv)
  ray_y = np.sin(yv)
  ray_z = np.cos(yv) * np.cos(xv)

  ray_stereo = np.stack([ray_x, ray_y, ray_z], axis=0)
  ray_rig = np.einsum('ij,jhw->ihw', stereo2rig, ray_stereo)

  lon = np.arctan2(ray_rig[0], ray_rig[2])
  lat = np.arcsin(ray_rig[1])

  # Map to equirectangular coordinates
  u = (
      (lon - np.radians(meta_fov['start_yaw_in_degrees']))
      / np.radians(
          meta_fov['end_yaw_in_degrees'] - meta_fov['start_yaw_in_degrees']
      )
      * (equirect_hw[1] - 1)
  )
  v = (
      (lat - np.radians(meta_fov['start_tilt_in_degrees']))
      / np.radians(
          meta_fov['end_tilt_in_degrees'] - meta_fov['start_tilt_in_degrees']
      )
      * (equirect_hw[0] - 1)
  )

  # Flatten arrays
  map_x = u.astype(np.float32)
  map_y = v.astype(np.float32)
  return map_x, map_y

def rectify_equirect_frame(image, meta_fov, corrections):
  """
  Rectifies an equirectangular frame based on rig calibration results (corrections).
  First split it into left and right images,
  applying rectification maps to each half, and then concatenating the results.

  Args:
    image (np.ndarray): The input equirectangular image to be rectified.
    meta_fov (float): The field of view metadata used for rectification.
    corrections (dict): A dictionary containing rig calibration for
              the left and right images.

  Returns:
    np.ndarray: The rectified equirectangular image.
  """
  left = image[:, : image.shape[1] // 2]
  right = image[:, image.shape[1] // 2 :]
  xx, yy = get_equirect_rectification_map(
      left.shape[:2], meta_fov, corrections['stereo2rig_left']
  )
  rect_equirect_left = cv2.remap(
      left,
      xx,
      yy,
      interpolation=cv2.INTER_LINEAR,
      borderMode=cv2.BORDER_CONSTANT,
      borderValue=(0, 0, 0),
  )
  xx, yy = get_equirect_rectification_map(
      right.shape[:2], meta_fov, corrections['stereo2rig_right']
  )
  rect_equirect_right = cv2.remap(
      right,
      xx,
      yy,
      interpolation=cv2.INTER_LINEAR,
      borderMode=cv2.BORDER_CONSTANT,
      borderValue=(0, 0, 0),
  )
  return np.concatenate([rect_equirect_left, rect_equirect_right], axis=1)


def rectify_equirect_frame_wrapper(kwargs):
    """
    Wrapper function to unpack kwargs and call the actual function.
    """
    return rectify_equirect_frame(**kwargs)

class NewCropFlags:
  output_hfov = 60
  imh = 512
  imw = 512
  meta_fov = None


def field_of_view_to_focal_length(fov_degrees: float, size: float) -> float:
  return size * 0.5 / np.tan(0.5 * (math.pi / 180.0) * fov_degrees)

def create_rotation_matrix(
    roll: float,
    pitch: float,
    yaw: float,
) -> np.ndarray:
  r"""Create c2w from euler angles.

  Args:
      roll (float): camera rotation about camera frame x-axis, in radians
      pitch (float): camera rotation about camera frame z-axis, in radians
      yaw (float): camera rotation about camera frame y-axis, in radians

  Returns:
      np.ndarray: rotation r_y @ r_x @ r_z (inv(extrinsics))
  """
  # calculate rotation about the x-axis
  r_x = np.array([
      [1.0, 0.0, 0.0],
      [0.0, np.cos(pitch), -np.sin(pitch)],
      [0.0, np.sin(pitch), np.cos(pitch)],
  ])
  # calculate rotation about the y-axis
  r_y = np.array([
      [np.cos(yaw), 0.0, np.sin(yaw)],
      [0.0, 1.0, 0.0],
      [-np.sin(yaw), 0.0, np.cos(yaw)],
  ])
  # calculate rotation about the z-axis
  r_z = np.array([
      [np.cos(roll), -np.sin(roll), 0.0],
      [np.sin(roll), np.cos(roll), 0.0],
      [0.0, 0.0, 1.0],
  ])

  return r_y @ r_x @ r_z

def create_jaxcam(
    roll: float,
    pitch: float,
    yaw: float,
    hfov: float,
    height: int,
    width: int,
    is_fisheye: bool,
):
  """Convert perspective camera parameters to a `jaxcam.Camera` object.

  Parameters:
  -----------
  roll, pitch, yaw : float
      Camera angles in degrees.
  height, width : int
      Dimensions of the output perspective image.
  hfov : float
      Horizontal field of view (FOV) in degrees.
  meta_fov : dict
      FOV parameters in degrees with keys:
      'start_yaw_in_degrees', 'end_yaw_in_degrees',
      'start_tilt_in_degrees', 'end_tilt_in_degrees'.

  Returns:
  --------
  camera : jaxcam.Camera
      A `jaxcam.Camera` object configured with the given perspective parameters.

  Notes:
  ------
  - The function computes the focal length from the horizontal field of view and
  image width.
  - The principal point is set to the center of the image.
  - The orientation matrix is created from the roll, pitch, and yaw angles.
  - The resulting `jaxcam.Camera` object can be used for various camera-related
  operations in the JAX environment.
  """
  principal_point = jnp.asarray([0.5 * width, 0.5 * height])
  if is_fisheye:
    assert width == height
    fx = width / np.radians(hfov)  # Focal length in pixels
  else:
    # Convert field of view to focal length
    fx = field_of_view_to_focal_length(hfov, width)
  # Create rotation matrix from roll, pitch, and yaw
  orientation = create_rotation_matrix(
      np.radians(roll), np.radians(pitch), np.radians(yaw)
  )
  # Create camera object
  camera = jaxcam.Camera.create(
      orientation=jnp.asarray(orientation.T),
      position=jnp.zeros(3),
      focal_length=jnp.asarray(fx),
      principal_point=principal_point,
      image_size=(jnp.asarray([width, height], dtype=jnp.float32)),
      pixel_aspect_ratio=1.0,
      radial_distortion=None,
      is_fisheye=is_fisheye,
  )
  return camera


def equirectangular_to_jaxcam_map(
    equirect_hw: tuple[int, int], meta_fov: dict, camera: jaxcam.Camera
):
  """Generate a UV map from a camera's perspective corresponding to an equirectangular image.

  Parameters:
  -----------
  equirect_hw : tuple[int, int]
      The height and width of the equirectangular image as a tuple (height,
      width).

  meta_fov : dict
      A dictionary containing the field of view (FOV) parameters in degrees. The
      dictionary should have the following keys:
      - 'start_yaw_in_degrees' : float
          The starting yaw angle in degrees.
      - 'end_yaw_in_degrees' : float
          The ending yaw angle in degrees.
      - 'start_tilt_in_degrees' : float
          The starting tilt angle (pitch) in degrees.
      - 'end_tilt_in_degrees' : float
          The ending tilt angle (pitch) in degrees.

  camera : jaxcam.Camera
      A `jaxcam.Camera` object containing camera parameters including the image
      size.

  Returns:
  --------
  map_x : np.ndarray
      A 2D array of x-coordinates in the equirectangular image corresponding to
      the camera's perspective.

  map_y : np.ndarray
      A 2D array of y-coordinates in the equirectangular image corresponding to
      the camera's perspective.

  Notes:
  ------
  - The function creates a UV map that can be used to transform an
  equirectangular image into a perspective view based on the camera parameters.
  - The camera parameters are assumed to be provided by the `jaxcam.Camera`
  object.
  - The function uses numpy for calculations and meshgrid creation.
  - The resulting UV map (map_x, map_y) can be used for sampling pixels from the
  equirectangular image to create the perspective image.
  """
  width, height = camera.image_size.astype(int)
  # Create coordinate grid
  x = np.linspace(0, width - 1, width) + 0.5
  y = np.linspace(0, height - 1, height) + 0.5
  xv, yv = np.meshgrid(x, y)
  rays = jaxcam.pixels_to_rays(
      camera, np.stack([xv, yv], axis=-1), normalize=True
  )

  ray_x = rays[..., 0]
  ray_y = rays[..., 1]
  ray_z = rays[..., 2]

  lon = np.arctan2(ray_x, ray_z)
  lat = np.arcsin(ray_y)

  # Map to equirectangular coordinates
  u = (
      (lon - np.radians(meta_fov['start_yaw_in_degrees']))
      / np.radians(
          meta_fov['end_yaw_in_degrees'] - meta_fov['start_yaw_in_degrees']
      )
      * (equirect_hw[1] - 1)
  )
  v = (
      (lat - np.radians(meta_fov['start_tilt_in_degrees']))
      / np.radians(
          meta_fov['end_tilt_in_degrees'] - meta_fov['start_tilt_in_degrees']
      )
      * (equirect_hw[0] - 1)
  )

  # Flatten arrays
  map_x = u.astype(np.float32)
  map_y = v.astype(np.float32)
  return map_x, map_y

def equirect_to_pers(
    equirect_img: np.ndarray,
    height: int,
    width: int,
    roll: float,
    pitch: float,
    yaw: float,
    hfov: float,
    meta_fov: dict,
):
  """Convert an equirectangular image to a perspective image.

  Parameters:
  -----------
  equirect_img : np.ndarray
      Input equirectangular image.
  height, width : int
      Dimensions of the output perspective image.
  roll, pitch, yaw : float
      Camera angles in degrees.
  hfov : float
      Horizontal field of view (FOV) in degrees.
  meta_fov : dict
      FOV parameters in degrees with keys:
      'start_yaw_in_degrees', 'end_yaw_in_degrees',
      'start_tilt_in_degrees', 'end_tilt_in_degrees'.

  Returns:
  --------
  pers_img : np.ndarray
      Output perspective image.

  Notes:
  ------
  - Creates a perspective camera with given parameters.
  - Maps the equirectangular image to the perspective view.
  - Uses OpenCV remap for the final perspective image.
  """
  camera = create_jaxcam(
      roll, pitch, yaw, hfov, height, width, is_fisheye=False
  )
  xx, yy = equirectangular_to_jaxcam_map(
      equirect_img.shape[:2], meta_fov, camera
  )
  pers_img = cv2.remap(
      equirect_img,
      xx,
      yy,
      interpolation=cv2.INTER_LINEAR,
      borderMode=cv2.BORDER_CONSTANT,
      borderValue=(0, 0, 0),
  )
  return pers_img

def rectified_equirect_left_right_crop_to_perspective(
    rectified_equirect_left_right: np.ndarray, 
    crop_flag: NewCropFlags
):

  """
  Given a rectified equirectangular left and right image, crop to perspective.

  Args:
      rectified_equirect_left_right (numpy.ndarray): A rectified equirectangular image containing both left and right images side by side.
      crop_flag (object): An object containing cropping parameters including:
          - imh (int): Image height for the perspective crop.
          - imw (int): Image width for the perspective crop.
          - output_hfov (float): Horizontal field of view for the output perspective image.
          - meta_fov (dict): Field of view of equirectangular.

  Returns:
      tuple: A tuple containing:
          - left_perspective (numpy.ndarray): The perspective-cropped left image.
          - right_perspective (numpy.ndarray): The perspective-cropped right image.
  """
  left = rectified_equirect_left_right[
      :, : rectified_equirect_left_right.shape[1] // 2
  ]
  right = rectified_equirect_left_right[
      :, rectified_equirect_left_right.shape[1] // 2 :
  ]
  left_perspective = equirect_to_pers(
      left,
      crop_flag.imh,
      crop_flag.imw,
      0,
      0,
      0,
      crop_flag.output_hfov,
      crop_flag.meta_fov,
  )
  right_perspective = equirect_to_pers(
      right,
      crop_flag.imh,
      crop_flag.imw,
      0,
      0,
      0,
      crop_flag.output_hfov,
      crop_flag.meta_fov,
  )
  return left_perspective, right_perspective

def rectified_equirect_left_right_crop_to_perspective_wrapper(kwargs):
    """
    Wrapper function to unpack kwargs and call the actual function.
    """
    return rectified_equirect_left_right_crop_to_perspective(**kwargs)



def load_rectified_video(video_id: str, clip_id: int, output_dir: str, raw_video_folder: str):
  """
  Load and rectify a video to perspective frames.
  This function performs the following steps:
  1. Load rig calibration data.
  2. Load equirectangular video frames.
  3. Rectify the equirectangular frames to perspective frames.
  4. Save the rectified equirectangular video.
  5. Crop the rectified equirectangular frames to perspective images.
  6. Save the rectified perspective frames.
  Args:
    video_id (str): The ID of the video to be processed.
    clip_id (int): The ID of the clip to be processed.
    output_dir (str): The directory where the output videos will be saved.
    raw_video_folder (str): The folder containing the raw video files.
  """
  vid = f'{video_id}-clip{clip_id}'
  os.makedirs(osp.join(output_dir, vid), exist_ok=True)
  with open('release_test.json', 'r') as f:
    metas = json.load(f)
  meta_fov = metas[video_id][clip_id]['meta_fov']
  corrections = {k: np.array(v) for k, v in metas[video_id][clip_id]['corrections'].items()}
  
  # Load video
  timestamps = metas[video_id][clip_id]['timestamp']
  equi_loader = EquiVideoLoader(video_id, raw_video_folder)
  # equi_video = equi_loader.retrieve_frame_cv2(timestamps)
  equi_video = equi_loader.retrieve_frames_moviepy(timestamps)
  media.write_video(
    osp.join(output_dir, vid, f"{vid}-raw_equirect.mp4"),
    equi_video, 
    fps=30
  )
  equi_video = media.read_video(osp.join(output_dir, vid, f"{vid}-raw_equirect.mp4"))

  rectified_equi_video = process_map(
    rectify_equirect_frame_wrapper,
    [
      {'image': x, 'meta_fov': meta_fov, 'corrections': corrections}
      for x in equi_video
    ],
    max_workers=4,
    desc='Rectify equirect frames',
  )
  rectified_equi_video = np.stack(rectified_equi_video, axis=0)
  media.write_video(
    osp.join(output_dir, f'{video_id}-clip{clip_id}', f"{video_id}-clip{clip_id}-rectified_equirect.mp4"),
    rectified_equi_video, 
    fps=30
  )
  rectified_equi_video = media.read_video(osp.join(output_dir, f'{video_id}-clip{clip_id}', f"{video_id}-clip{clip_id}-rectified_equirect.mp4"))
  logging.info('Recropping from equirectangular')
  crop_flag = NewCropFlags()
  crop_flag.meta_fov = meta_fov
  
  output = process_map(
    rectified_equirect_left_right_crop_to_perspective_wrapper,
    [
      {'rectified_equirect_left_right': x, 'crop_flag': crop_flag}
      for x in rectified_equi_video
    ],
    max_workers=8,
    desc="Crop to perspective images"
  )

  input_stereo = {'left': {}, 'right': {}}
  input_stereo['left']['video'] = np.stack(
    [pair[0] for pair in output], axis=0
  )
  input_stereo['right']['video'] = np.stack(
      [pair[1] for pair in output], axis=0
  )

  logging.info('Saving rectified perspective frames')
  for key in ['left', 'right']:
    media.write_video(
        osp.join(
            output_dir,
            vid,
            f'{vid}-{key}_rectified.mp4',
        ),
        input_stereo[key]['video'],
        fps=30,
    )

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--videoid', help='video id', type=str, default='')
  parser.add_argument('--clipid', help='clip id', type=int, default=0)
  parser.add_argument('--raw_video_folder', help='raw video folder', type=str, default='stereo4d_dataset/raw')
  parser.add_argument('--output_folder', help='output folder', type=str, default='')

  args = parser.parse_args()

  load_rectified_video(args.videoid, args.clipid, args.output_folder, args.raw_video_folder)

if __name__ == '__main__':
  main()
