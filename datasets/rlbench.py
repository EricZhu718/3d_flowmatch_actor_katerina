import json
import random

from .base import BaseDataset
from torch.utils.data import Dataset
import os, torchvision, torch.nn.functional as F, torch
import numpy as np

PERACT_TASKS = [
    "place_cups", "close_jar", "insert_onto_square_peg",
    "light_bulb_in", "meat_off_grill", "open_drawer",
    "place_shape_in_shape_sorter", "place_wine_at_rack_location",
    "push_buttons", "put_groceries_in_cupboard",
    "put_item_in_drawer", "put_money_in_safe", "reach_and_drag",
    "slide_block_to_color_target", "stack_blocks", "stack_cups",
    "sweep_to_dustpan_of_size", "turn_tap"
]
PERACT2_TASKS = [
    'bimanual_push_box',
    'bimanual_lift_ball',
    'bimanual_dual_push_buttons',
    'bimanual_pick_plate',
    'bimanual_put_item_in_drawer',
    'bimanual_put_bottle_in_fridge',
    'bimanual_handover_item',
    'bimanual_pick_laptop',
    'bimanual_straighten_rope',
    'bimanual_sweep_to_dustpan',
    'bimanual_lift_tray',
    'bimanual_handover_item_easy',
    'bimanual_take_tray_out_of_oven'
]

import numpy as np

import numpy as np

def batch_rpy_to_xyzw_np(poses: np.ndarray, normalize: bool = True) -> np.ndarray:
    assert poses.ndim == 2 and poses.shape[-1] == 6, "Expected shape (N, 6)."
    xyz = poses[:, :3]
    roll, pitch, yaw = poses[:, 3], poses[:, 4], poses[:, 5]

    cr, sr = np.cos(roll * 0.5), np.sin(roll * 0.5)
    cp, sp = np.cos(pitch * 0.5), np.sin(pitch * 0.5)
    cy, sy = np.cos(yaw * 0.5), np.sin(yaw * 0.5)

    w = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    quat = np.stack([qx, qy, qz, w], axis=-1)

    if normalize:
        quat = quat / (np.linalg.norm(quat, axis=-1, keepdims=True) + 1e-12)

    return np.concatenate([xyz, quat], axis=-1)

class EgoDexDataset(BaseDataset):
    """EgoDexDataset dataset."""
    quat_format= 'xyzw'
    def __init__(
        self,
        root,
        instructions,
        copies=None,
        relative_action=False,
        mem_limit=8,
        actions_only=False,
        chunk_size=4
    ):
        # self.copies = self.train_copies if copies is None else copies
        self._relative_action = relative_action
        self._actions_only = actions_only
        self.chunk_size = chunk_size
        self.n_cam = 1 # egodex only uses 1 camera
        
        # Load all annotations manually
        RESIZED_IMG_SIZE = 256 # resize the images to 256 by 256 because I believe that's what 
        # RLBench's preprocessor assumes they are by default in this codebase

        self.annos = {
            'action': [],
            'depth': [],
            'rgb' : [],
            'extrinsics' : [],
            'intrinsic' : [],
            'task' : [],
            'instruction' : []
        }
        # for task in os.listdir(root):
        #     for run in os.listdir(task):
        for run in os.listdir(root):
                task = root.name # since task is not specified in EgoDex, I assume that the name of the directory is the task name
                
                DEPTH_SCALE = 50

                # load frames and depth images
                frames, _, _ = torchvision.io.read_video(os.path.join(root, run, "rgb_frames.mp4"))
                depth_frames, _, _ = torchvision.io.read_video(os.path.join(root, run, "depth_frames.mp4"))
                depth_frames = depth_frames / DEPTH_SCALE # rescale the depth

                original_width, original_height = frames.shape[2], frames.shape[1]
                
                # resize both depth and rgb frames to 256 by 256, or whatever IMG_SIZE is
                frames = frames.permute(0, 3, 1, 2).float()  # (N, C, H, W)
                frames = F.interpolate(frames, size=(RESIZED_IMG_SIZE, RESIZED_IMG_SIZE), mode="bilinear", align_corners=False).numpy()
                depth_frames = depth_frames.permute(0, 3, 1, 2).float()  # (N, C, H, W)
                depth_frames = F.interpolate(depth_frames, size=(RESIZED_IMG_SIZE, RESIZED_IMG_SIZE), mode="bilinear", align_corners=False).numpy()[:,0,:,:] # remove the channel component because each image is a stack of 3 grey scaled images

                # load metadata with all the actions and camera matricies
                with open(os.path.join(root, run, "data.json")) as fp:
                    metadata = json.load(fp)
                
                instrinsic = np.array(metadata['instrinsic']) # [N, 3, 3]
                # need to change intrinsic matrix because I have rescaled the images, which changes the intrinsic matrix
                instrinsic[0] = instrinsic[0] * RESIZED_IMG_SIZE / original_width
                instrinsic[1] = instrinsic[1] * RESIZED_IMG_SIZE / original_height

                extrinsics = np.array(metadata['extrinsics']) # [N, 4, 4]
                left_hand_xyzrpy = np.array(metadata['left_hand_xyzrpy']) # [N, 6]
                left_hand_xyz_and_quat = batch_rpy_to_xyzw_np(left_hand_xyzrpy)
                right_hand_xyzrpy = np.array(metadata['right_hand_xyzrpy']) # [N, 6]
                right_hand_xyz_and_quat = batch_rpy_to_xyzw_np(right_hand_xyzrpy)

                KEEP_LAST_NUM_FRAMES = 50
                for frame_idx in range(len(frames))[len(frames)-KEEP_LAST_NUM_FRAMES:len(frames)]: # Only take hte last 50 frames (need to make all trajectories the same length, and they are not in egodex)
                    # right now, all values are numpy arrays except task and instruction which are strings

                    # get the xyz_quat so now the array should be of size (T, 2, 7)
                    actions_xyz_quat = np.stack([left_hand_xyz_and_quat[len(frames)-KEEP_LAST_NUM_FRAMES:len(frames)], right_hand_xyz_and_quat[len(frames)-KEEP_LAST_NUM_FRAMES:len(frames)]],axis=1)
                    
                    # add a dummy gripper close state into the action. This makes the action vector go from 7 to 8. It was not possible to infer hand closed / open from the egodex dataset
                    actions_xyz_quat_and_close = np.concat([actions_xyz_quat, np.zeros(shape=actions_xyz_quat.shape[:2]+(1,))], axis=2) # (T, 2, 8)

                    self.annos['action'].append(actions_xyz_quat_and_close) # keep the trajectory, but only the KEEP_LAST_NUM_FRAMES of steps
                    self.annos['depth'].append(depth_frames[frame_idx])
                    self.annos['rgb'].append(frames[frame_idx])
                    self.annos['extrinsics'].append(extrinsics[frame_idx])
                    self.annos['intrinsic'].append(instrinsic)
                    self.annos['task'].append(metadata['task'])
                    self.annos['instruction'].append(metadata['instruction'])
        

                    

        # Sanity check
        len_ = len(self.annos['action'])
        self.len_ = len_
        for key in self.annos:
            assert len(self.annos[key]) == len_, f'length mismatch in {key}'
        print(f"Found {len(self.annos['action'])} samples")
        
        # super().__init__(
        #     root=root,
        #     instructions=instructions,
        #     copies=copies,
        #     relative_action=relative_action,
        #     mem_limit=mem_limit,
        #     actions_only=actions_only,
        #     chunk_size=chunk_size
        # )

    def __len__(self):
        return self.len_

    def __getitem__(self, idx):
        """
        self.annos: {
            action: (N, T, 8) float
            depth: (N, n_cam, H, W) float16
            proprioception: (N, nhist, 8) float
            rgb: (N, n_cam, 3, H, W) uint8
            task_id: (N,) uint8
            variation: (N,) uint8
            extrinsics: (N, n_cam, 4, 4) float
            intrinsics: (N, n_cam, 3, 3) float
        }
        """
        # First detect which copy we fall into
        idx = idx % (len(self.annos['action']) // self.chunk_size)
        # and then which chunk
        idx = idx * self.chunk_size
        if self._actions_only:
            return {"action": self._get_action(idx)}
        return {
            # because task is not given in egodex, assume task to be name of task in folder
            "task": self.annos['task'][idx:idx+self.chunk_size],  # [str]

            # because instruction is not given in egodex, assume instruction to be same name as task
            "instr": self.annos['instruction'][idx:idx+self.chunk_size],  # [str]

            # the [None, None, :] is to add a dummy dimension for the n_cam2d/n_cam3d and other because of the default collate function concatenates instead of stacks for a batch. In our setup, there is only 1 camera
            "rgb": torch.as_tensor(np.stack(self.annos['rgb'][idx])).clip(0, 255).to(torch.uint8).to(torch.uint8)[None, None,:].to(torch.float),  # tensor(1, n_cam3d, 3, H, W)
            "depth": torch.as_tensor(np.stack(self.annos['depth'][idx])).clip(0, 255).to(torch.uint8).to(torch.uint8)[None, None,:].to(torch.float),  # tensor(1, n_cam3d, H, W)
            # "rgb2d":  torch.as_tensor(np.stack(self.annos['rgb'][idx])).clip(0, 255).to(torch.uint8).to(torch.uint8)[None, None,:],  # tensor(1, n_cam2d, 3, H, W)
            "rgb2d" : None,
            "extrinsics": torch.as_tensor(self.annos['extrinsics'][idx])[None, None,:].to(torch.float),  # tensor(1, n_cam3d, 4, 4)
            "intrinsics": torch.as_tensor(self.annos['intrinsic'][idx])[None, None,:].to(torch.float),  # tensor(1, n_cam3d, 3, 3)
            "action": torch.as_tensor(self.annos['action'][idx])[None, :].to(torch.float),  # tensor(1, T, nhand, 8)

            # no proprioception is given in egodex, so I added dummy zero tensor, the 3 is because the num_history is 3. In a real RLBench dataset, it would already have the shape (1,3,8)
            "proprioception": torch.tensor([[[0.]*8]*2]*3)[None,:].to(torch.float),  # tensor(1, 8)

        }

class RLBenchDataset(BaseDataset):
    """RLBench dataset."""
    quat_format= 'xyzw'

    def __init__(
        self,
        root,
        instructions,
        copies=None,
        relative_action=False,
        mem_limit=8,
        actions_only=False,
        chunk_size=4
    ):
        super().__init__(
            root=root,
            instructions=instructions,
            copies=copies,
            relative_action=relative_action,
            mem_limit=mem_limit,
            actions_only=actions_only,
            chunk_size=chunk_size
        )

    def _get_task(self, idx):
        return [
            self.tasks[int(tid)]
            for tid in self.annos['task_id'][idx:idx + self.chunk_size]
        ]

    def _get_instr(self, idx):
        return [
            random.choice(self._instructions[self.tasks[int(t)]][str(int(v))])
            for t, v in zip(
                self.annos['task_id'][idx:idx + self.chunk_size],
                self.annos['variation'][idx:idx + self.chunk_size]
            )
        ]

    def _get_rgb2d(self, idx):
        if self.camera_inds2d is not None:
            return self._get_attr_by_idx(idx, 'rgb', False)[:, self.camera_inds2d]
        return None

    def _get_extrinsics(self, idx):
        return self._get_attr_by_idx(idx, 'extrinsics', True)

    def _get_intrinsics(self, idx):
        return self._get_attr_by_idx(idx, 'intrinsics', True)

    def __getitem__(self, idx):
        """
        self.annos: {
            action: (N, T, 8) float
            depth: (N, n_cam, H, W) float16
            proprioception: (N, nhist, 8) float
            rgb: (N, n_cam, 3, H, W) uint8
            task_id: (N,) uint8
            variation: (N,) uint8
            extrinsics: (N, n_cam, 4, 4) float
            intrinsics: (N, n_cam, 3, 3) float
        }
        """
        # First detect which copy we fall into
        idx = idx % (len(self.annos['action']) // self.chunk_size)
        # and then which chunk
        idx = idx * self.chunk_size
        if self._actions_only:
            return {"action": self._get_action(idx)}
        return {
            "task": self._get_task(idx),  # [str]
            "instr": self._get_instr(idx),  # [str]
            "rgb": self._get_rgb(idx),  # tensor(n_cam3d, 3, H, W)
            "depth": self._get_depth(idx),  # tensor(n_cam3d, H, W)
            "rgb2d": self._get_rgb2d(idx),  # tensor(n_cam2d, 3, H, W)
            "proprioception": self._get_proprioception(idx),  # tensor(1, 8)
            "action": self._get_action(idx),  # tensor(T, 8)
            "extrinsics": self._get_extrinsics(idx),  # tensor(n_cam3d, 4, 4)
            "intrinsics": self._get_intrinsics(idx)  # tensor(n_cam3d, 3, 3)
        }


class HiveformerDataset(RLBenchDataset):
    cameras = ("wrist", "front")
    camera_inds = None
    train_copies = 100
    camera_inds2d = None

    def _load_instructions(self, instruction_file):
        instr = json.load(open(instruction_file))
        self.tasks = list(instr.keys())
        return instr


class PeractDataset(RLBenchDataset):
    """RLBench dataset under Peract setup."""
    tasks = PERACT_TASKS
    cameras = ("left_shoulder", "right_shoulder", "wrist", "front")
    camera_inds = None
    train_copies = 10
    camera_inds2d = None

    def __getitem__(self, idx):
        """
        self.annos: {
            action: (N, T, 8) float
            depth: (N, n_cam, H, W) float16
            proprioception: (N, nhist, 8) float
            rgb: (N, n_cam, 3, H, W) uint8
            task_id: (N,) uint8
            variation: (N,) uint8
            extrinsics: (N, n_cam, 4, 4) float
            intrinsics: (N, n_cam, 3, 3) float
        }
        """
        # First detect which copy we fall into
        idx = idx % (len(self.annos['action']) // self.chunk_size)
        # and then which chunk
        idx = idx * self.chunk_size
        if self._actions_only:
            return {"action": self._get_action(idx)}
        return {
            "task": self._get_task(idx),  # [str]
            "instr": self._get_instr(idx),  # [str]
            "rgb": self._get_rgb(idx),  # tensor(n_cam3d, 3, H, W)
            "pcd": self._get_attr_by_idx(idx, 'pcd', True),  # tensor(n_cam3d, H, W)
            "proprioception": self._get_proprioception(idx),  # tensor(1, 8)
            "action": self._get_action(idx),  # tensor(T, 8)
        }


class PeractTwoCamDataset(PeractDataset):
    """RLBench dataset under Peract setup."""
    tasks = PERACT_TASKS
    cameras = ("wrist", "front")
    camera_inds = [2, 3]
    train_copies = 10
    camera_inds2d = None


class Peract2Dataset(RLBenchDataset):
    """RLBench dataset under Peract2 setup."""
    tasks = PERACT2_TASKS
    cameras = ("front", "wrist_left", "wrist_right")
    camera_inds = None
    train_copies = 10
    camera_inds2d = None


class Peract2SingleCamDataset(RLBenchDataset):
    """RLBench dataset under Peract2 setup."""
    tasks = PERACT2_TASKS
    cameras = ("front",)
    camera_inds = (0,)  # use only front camera
    train_copies = 10
    camera_inds2d = None
