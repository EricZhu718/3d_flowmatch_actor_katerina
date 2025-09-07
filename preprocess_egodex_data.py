import h5py
import torchvision, torch
import pathlib
import numpy as np, math
from PIL import Image
import os, json
import psutil, os
import gc

from transformers import AutoImageProcessor, GLPNForDepthEstimation  # direct model path

def convert_transform_mats_to_xyzrpy(T_matrices): # convert from 4x4 matrix to xyzrpy
    T = np.asarray(T_matrices, dtype=float)
    assert T.ndim == 3 and T.shape[1:] == (4, 4), "Input must be (N, 4, 4)"
    N = T.shape[0]
    results = np.zeros((N, 6), dtype=float)
    for i in range(N):
        Ti = T[i]
        x, y, z = Ti[0, 3], Ti[1, 3], Ti[2, 3]
        r00, r01, r02 = Ti[0, 0], Ti[0, 1], Ti[0, 2]
        r10, r11, r12 = Ti[1, 0], Ti[1, 1], Ti[1, 2]
        r20, r21, r22 = Ti[2, 0], Ti[2, 1], Ti[2, 2]
        r20_clamped = np.clip(r20, -1.0, 1.0)
        if abs(r20_clamped) < 1.0 - 1e-9:
            pitch = -math.asin(r20_clamped)
            roll  = math.atan2(r21, r22)
            yaw   = math.atan2(r10, r00)
        else:
            pitch = math.pi/2 if r20_clamped <= -1.0 + 1e-9 else -math.pi/2
            roll  = 0.0
            yaw   = math.atan2(-r01, r11)
        results[i] = [x, y, z, roll, pitch, yaw]
    return results

if __name__ == "__main__":
    path_to_dataset = "ego_dex_sample_dataset"
    path_to_post_processed_dir = "ego_dex_processed_dataset"
    model_id = "vinvino02/glpn-nyu"
    DEPTH_SCALE = 50
    BATCH_SIZE = 8 # used for batch decoding, I used 8 cause that's fine for the gpu I was using

    # initialize the depth processor, I used one online that seemed decent
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.float16 if device == "cuda" else torch.float32
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = GLPNForDepthEstimation.from_pretrained(model_id).to(device=device, dtype=dtype).eval()

    os.mkdir(path_to_post_processed_dir)

    for task in os.listdir(path_to_dataset): # for each task in the egodex dataset
        os.mkdir(os.path.join(path_to_post_processed_dir, task))
        num_videos = len(os.listdir(os.path.join(path_to_dataset, task))) // 2

        for video_idx in range(num_videos): # for each video of a task

            # this is to track memory, I added it because there was a memory leak at one point, but it should be fixed now
            process = psutil.Process(os.getpid())
            print(f"Memory used: {process.memory_info().rss / 1024 ** 2:.2f} MB")

            out_dir = os.path.join(path_to_post_processed_dir, task, str(video_idx))
            os.mkdir(out_dir)

            # read HDF5 file and extract the right hand and left hand transformation matricies
            with h5py.File(str(pathlib.Path(path_to_dataset) / task / f"{video_idx}.hdf5"), "r") as f:
                intrinsic = np.array(f['camera']['intrinsic'])
                extrinsics = np.array(f['transforms']['camera'])
                right_hand_transforms = np.array(f['transforms']['rightHand'])
                left_hand_transforms  = np.array(f['transforms']['leftHand'])

            # RL bench uses poses in the xyz rpy, so i convert them to that form right now
            right_hand_xyzrpy = convert_transform_mats_to_xyzrpy(right_hand_transforms)
            left_hand_xyzrpy  = convert_transform_mats_to_xyzrpy(left_hand_transforms)
            del right_hand_transforms, left_hand_transforms # delete because there seems to be a memory leak somewhere

            # read RGB frames
            frames, _, _ = torchvision.io.read_video(str(pathlib.Path(path_to_dataset) / task / f"{video_idx}.mp4"))
            # frames = frames[:20]  # random debug I want

            # convert to pil images just cause that's what hte processor wantsx
            frames_pil = [Image.fromarray(frames[i].numpy()) for i in range(frames.size(0))]

            depth_pred = []  # list of (H,W) float32
            with torch.inference_mode():
                for start in range(0, len(frames_pil), BATCH_SIZE): # do batch computing of the depth estimator, if CUDA crashes, lower batch_size
                    batch_pils = frames_pil[start:start + BATCH_SIZE]
                    inputs = processor(images=batch_pils, return_tensors="pt")
                    # move only tensors to device
                    inputs = {k: (v.to(device=device, dtype=dtype) if hasattr(v, "to") else v)
                              for k, v in inputs.items()}
                    outputs = model(**inputs)  # predicted_depth: [B,1,H,W]
                    preds = outputs.predicted_depth.squeeze(1).float().cpu().numpy()  # (B,H,W)
                    for d in preds:
                        depth_pred.append(d * DEPTH_SCALE)
                    del inputs, outputs, preds, batch_pils
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            # write RGB video
            torchvision.io.write_video(os.path.join(out_dir, "rgb_frames.mp4"), frames, fps=30)

            # write depth video
            depth_np = np.stack(depth_pred, axis=0)                 # (T,H,W)
            depth_np = np.clip(depth_np, 0, 255).astype(np.uint8)   # uint8
            depth_np = np.repeat(depth_np[..., None], 3, axis=-1)   # (T,H,W,3)
            depth_t  = torch.from_numpy(depth_np)                   # zero-copy
            torchvision.io.write_video(os.path.join(out_dir, "depth_frames.mp4"), depth_t, fps=30)

            # scale from 0 to 255 for better visualization
            depth_t = ((depth_t - depth_t.min()) / (depth_t - depth_t.min()).max() * 255).clip(0, 255).to(torch.uint8)
            torchvision.io.write_video(os.path.join(out_dir, "depth_visualization.mp4"), depth_t, fps=30)

            # keep rest of info as metadata JSON
            instruction = task.replace("_", " ")
            data_json = {
                'task': instruction,
                'instruction': instruction,
                'right_hand_xyzrpy': right_hand_xyzrpy.tolist(),
                'left_hand_xyzrpy': left_hand_xyzrpy.tolist(),
                'instrinsic': intrinsic.tolist(),
                'extrinsics': extrinsics.tolist(),
            }
            with open(os.path.join(out_dir, "data.json"), "w") as json_f:
                json.dump(data_json, json_f, indent=2)

            # cleanup per video gc usually does this implictly, but I might as well do it anyways to be safe
            del frames, frames_pil, depth_pred, depth_np, depth_t, data_json
            del right_hand_xyzrpy, left_hand_xyzrpy, intrinsic, extrinsics
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
