import sys
import os

# Add the SEA-RAFT/core directory to sys.path
sea_raft_core_path = os.path.abspath('./SEA-RAFT')
sys.path.append(sea_raft_core_path)

import argparse

import torch
import torch.nn.functional as F
import pickle

import mediapy as media
from core.raft import RAFT
import tqdm
import os.path as osp
import numpy as np

import json
import argparse

def json_to_args(json_path):
  # return a argparse.Namespace object
  with open(json_path, 'r') as f:
    data = json.load(f)
  args = argparse.Namespace()
  args_dict = args.__dict__
  for key, value in data.items():
    args_dict[key] = value
  return args

def parse_args(parser):
  entry = parser.parse_args()
  json_path = entry.cfg
  args = json_to_args(json_path)
  args_dict = args.__dict__
  for index, (key, value) in enumerate(vars(entry).items()):
    args_dict[key] = value
  return args

def forward_flow(args, model, image1, image2):
  output = model(image1, image2, iters=args.iters, test_mode=True)
  flow_final = output['flow'][-1]
  info_final = output['info'][-1]
  return flow_final, info_final

def calc_flow(args, model, image1, image2):
  img1 = F.interpolate(image1, scale_factor=2 ** args.scale, mode='bilinear', align_corners=False)
  img2 = F.interpolate(image2, scale_factor=2 ** args.scale, mode='bilinear', align_corners=False)
  H, W = img1.shape[2:]
  flow, info = forward_flow(args, model, img1, img2)
  flow_down = F.interpolate(flow, scale_factor=0.5 ** args.scale, mode='bilinear', align_corners=False) * (0.5 ** args.scale)
  info_down = F.interpolate(info, scale_factor=0.5 ** args.scale, mode='area')
  return flow_down, info_down

@torch.no_grad()
def raft_fn(left, right, args, model):
  flow_fwd, info_fwd = calc_flow(args, model, left, right)
  flow_bwd, info_bwd = calc_flow(args, model, right, left)
  flow_fwd = flow_fwd[0].permute(1, 2, 0).cpu().numpy()
  flow_bwd = flow_bwd[0].permute(1, 2, 0).cpu().numpy()
  flow = {
    'fwd': flow_fwd,
    'bwd': flow_bwd,
  }
  return flow


@torch.no_grad()
def inference_stereo_depth(model, args, device=torch.device('cuda')):
  vid = f"{args.videoid}-clip{args.clipid}"
  video1 = media.read_video(osp.join(args.output_folder, vid, vid + '-left_rectified.mp4'))
  video2 = media.read_video(osp.join(args.output_folder, vid, vid + '-right_rectified.mp4'))
  assert len(video1) == len(video2)
  flows = []
  for fid in tqdm.tqdm(range(len(video1))):
    image1 = video1[fid]
    image2 = video2[fid]
    image1 = torch.tensor(image1, dtype=torch.float32).permute(2, 0, 1)
    image2 = torch.tensor(image2, dtype=torch.float32).permute(2, 0, 1)
    H, W = image1.shape[1:]
    image1 = image1[None].to(device)
    image2 = image2[None].to(device)
    flows.append(raft_fn(image1, image2, args, model))

  with open(
    osp.join(
      osp.join(args.output_folder, vid, vid + '-flows_stereo.pkl')
    ),
    'wb',
  ) as f:
    pickle.dump(flows, f)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--cfg', help='experiment configure file name', default='SEA-RAFT/config/eval/spring-M.json', type=str)
  parser.add_argument('--path', help='checkpoint path', type=str, default=None)
  parser.add_argument('--url', help='checkpoint url', type=str, default='MemorySlices/Tartan-C-T-TSKH-spring540x960-M')
  parser.add_argument('--device', help='inference device', type=str, default='cpu')
  parser.add_argument('--videoid', help='video id', type=str, default='')
  parser.add_argument('--clipid', help='clip id', type=int, default=0)
  parser.add_argument('--output_folder', help='output folder', type=str, default='')

  args = parse_args(parser)
  if args.path is None and args.url is None:
    raise ValueError("Either --path or --url must be provided")
  model = RAFT.from_pretrained(args.url, args=args)
      
  if args.device == 'cuda':
    device = torch.device('cuda')
  else:
    device = torch.device('cpu')
  model = model.to(device)
  model.eval()
  inference_stereo_depth(model, args, device=device)

if __name__ == '__main__':
  main()