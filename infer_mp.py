#!/usr/bin/env python3
"""
VidTok: Video to Tokens Converter

Usage:
    python infer_mp.py --model kl_noncausal --video assets/example.mp4 --sample_fps 25 --output_dir ./output
    python infer_mp.py --model kl_noncausal --video dir --sample_fps 25 --output_dir ./output
    python infer.py --model kl_noncausal --series_dir /nfs/yanzhang.ljx/workspace/datasets/YingShi/clean/zh --sample_fps 25
Models: fsq, kl_noncausal, kl_causal, kl_causal_288
-----------------------------------------------------
添加 --decode 以解码视频隐空间特征，特征维度 [Batch, Channels, Time, Height, Width]；
-----------------------------------------------------
"""
import os
import sys
import argparse
import time
import torch
import decord
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.io import write_video
import torch.multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
mp.set_start_method('spawn', force=True)
sys.path.append(os.getcwd())
decord.bridge.set_bridge("torch")

from scripts.inference_evaluate import load_model_from_config
from vidtok.modules.util import compute_psnr, compute_ssim

# Model configurations
MODEL_CONFIGS = {
    "fsq": {
        "cfg": "configs/vidtok_fsq_noncausal_488_262144.yaml",
        "ckpt": "checkpoints/vidtok_fsq_noncausal_488_262144.ckpt",
        "desc": "FSQ Non-causal (Discrete)",
    },
    "kl_noncausal": {
        "cfg": "configs/vidtok_kl_noncausal_488_16chn.yaml",
        "ckpt": "checkpoints/vidtok_kl_noncausal_488_16chn.ckpt",
        "desc": "KL Non-causal (Continuous, best quality)",
    },
    "kl_causal": {
        "cfg": "configs/vidtok_kl_causal_488_16chn.yaml",
        "ckpt": "checkpoints/vidtok_kl_causal_488_16chn.ckpt",
        "desc": "KL Causal 488 (Continuous, streaming)",
    }
}


class VideoDataset(Dataset):
    def __init__(
        self,
        video_path: str,
        input_height: int,
        input_width: int,
        sample_fps: int = 25,
        chunk_size: int = 16,
        is_causal: bool = True,
        read_long_video: bool = True,
        num_chunks: int = 0,
    ):
        # 仅保存可序列化的基础类型，不在此处打开视频文件
        self.video_path = video_path
        self.input_height = input_height
        self.input_width = input_width
        self.sample_fps = sample_fps
        self.chunk_size = chunk_size
        self.is_causal = is_causal
        self.read_long_video = read_long_video
        self.num_chunks = num_chunks
        
        self.video_reader = None
        self.chunks = []
        self._initialized = False
        
        self.transform = transforms.Compose([
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ])

    def _lazy_init(self):
        """在 DataLoader Worker 进程中延迟初始化视频读取器"""
        if self._initialized:
            return
            
        self.video_reader = decord.VideoReader(
            self.video_path,
            width=self.input_width,
            height=self.input_height
        )
        total_frames = len(self.video_reader)
        fps = self.video_reader.get_avg_fps()

        interval = round(fps / self.sample_fps)
        frame_ids = list(range(0, total_frames, interval))
        video_length = len(frame_ids)

        if self.read_long_video:
            if self.is_causal:
                for x in range(0, video_length - 1, self.chunk_size):
                    end = min(x + self.chunk_size + 1, video_length)
                    chunk_frame_ids = frame_ids[x:end]
                    if len(chunk_frame_ids) == self.chunk_size + 1:
                        self.chunks.append(chunk_frame_ids)
            else:
                for x in range(0, video_length, self.chunk_size):
                    end = min(x + self.chunk_size, video_length)
                    chunk_frame_ids = frame_ids[x:end]
                    if len(chunk_frame_ids) == self.chunk_size:
                        self.chunks.append(chunk_frame_ids)
        else:
            num_frames_per_batch = self.chunk_size + 1 if self.is_causal else self.chunk_size
            for x in range(0, len(frame_ids), num_frames_per_batch):
                chunk_frame_ids = frame_ids[x : x + num_frames_per_batch]
                if len(chunk_frame_ids) == num_frames_per_batch:
                    self.chunks.append(chunk_frame_ids)
                    
        self._initialized = True

    def __len__(self):
        return self.num_chunks

    def __getitem__(self, idx):
        self._lazy_init()
        frame_ids = self.chunks[idx]
        frames = self.video_reader.get_batch(frame_ids)
        frames = frames.float() / 255.0
        frames = frames.permute(0, 3, 1, 2)
        frames = self.transform(frames)
        frames = frames.permute(1, 0, 2, 3)
        return frames, len(frame_ids)


def save_tokens(tokens: torch.Tensor, output_path: str, video_path: str, output_token_hz: float):
    """Save tokens to npy file."""
    data = {
        'tokens': tokens.cpu().numpy() if tokens is not None else None,
        'video_path': video_path,
        'output_token_hz': output_token_hz,
    }
    np.save(output_path, data, allow_pickle=True)
    print(f"Saved tokens: {output_path}")


def tensor_to_uint8(tensor: torch.Tensor) -> np.ndarray:
    """Convert tensor to uint8 video format."""
    tensor = tensor.permute(1, 2, 3, 0) # [C, T, H, W] to [T, H, W, C]
    tensor = torch.clamp(tensor, -1.0, 1.0)
    tensor = (tensor + 1.0) / 2.0
    return (tensor.cpu().numpy() * 255).astype(np.uint8) # [T, H, W, C]


def compute_metrics(original: torch.Tensor, reconstructed: torch.Tensor):
    """Compute PSNR and SSIM."""
    if original.dim() == 4:
        original = original.unsqueeze(0)
        reconstructed = reconstructed.unsqueeze(0)

    T_orig, T_recon = original.shape[2], reconstructed.shape[2]
    if T_orig != T_recon:
        T_min = min(T_orig, T_recon)
        print(f"  Warning: Input T={T_orig}, Output T={T_recon}, using T={T_min}")
        original = original[:, :, :T_min, :, :]
        reconstructed = reconstructed[:, :, :T_min, :, :]

    original = (original + 1.0) / 2.0
    reconstructed = (reconstructed + 1.0) / 2.0

    return compute_psnr(original, reconstructed).item(), compute_ssim(original, reconstructed).item()

def get_video_files_by_series(series_dir): # 根据实际路径更改次函数
    series_dir = Path(series_dir)
    series_pairs_map = {}
    # 遍历影视剧文件夹
    for series_folder in series_dir.iterdir():
        if not series_folder.is_dir():
            continue
        paired_paths = []
        clipped_dirs = []
        for sub_dir in series_folder.iterdir():
            if sub_dir.is_dir() and (sub_dir / "clipped").is_dir():
                clipped_dirs.append(sub_dir / "clipped")
        for clipped_path in clipped_dirs:
            # embs_video 目录
            embs_dir = clipped_path.parent / "embs_video"
            mp4_files = list(clipped_path.glob("*.mp4")) + list(clipped_path.glob("*.MP4"))
            for vid_path in mp4_files:
                # 保持原文件名 stem，替换为指定的输出扩展名
                out_path = embs_dir / f"{vid_path.stem}.npy"
                if out_path.is_file():  # 如果输出文件已存在，则避免重复处理
                    print(f"[INFO] already exists {out_path}")
                    continue
                paired_paths.append((vid_path, out_path))
        if not paired_paths:
            print(f"⚠️ Warning: No MP4 files found in any 'clipped' directory for series: {series_folder.name}")
            continue
        series_pairs_map[series_folder.name] = paired_paths
    print(f"Total series found: {len(series_pairs_map)}")
    return series_pairs_map


def scan_videos(video_files, series_videos_map, args, max_workers=200):
    print("\nScanning videos...")
    tasks = []
    if video_files:
        for vf in video_files:
            tasks.append((vf, Path(args.output_dir) / f"{Path(vf).stem}.npy", None))
    elif series_videos_map:
        for series_name, pairs_list in series_videos_map.items():
            for vid_path, out_path in pairs_list:
                tasks.append((vid_path, out_path, series_name))
                
    if not tasks:
        print("Error: No valid videos to process")
        return None

    def _extract_worker(inp_path, out_path, series_name):
        try:
            reader = decord.VideoReader(str(inp_path), num_threads=0)
            total_frames = len(reader)
            fps = reader.get_avg_fps()
            orig_h, orig_w = reader[0].shape[:2]
            del reader  # 立即释放句柄
            
            if args.adaptive_size:
                scale = min(args.max_size / orig_w, args.max_size / orig_h)
                tw = int(orig_w * scale) if scale < 1 else orig_w
                th = int(orig_h * scale) if scale < 1 else orig_h
                tw = ((tw - 1) // 8 + 1) * 8
                th = ((th - 1) // 8 + 1) * 8
            else:
                tw, th = 256, 256
                
            interval = round(fps / args.sample_fps)
            sampled = (total_frames - 1) // interval + 1
            
            return {
                'path': inp_path, 'output': out_path, 'series': series_name,
                'total_frames': total_frames, 'fps': fps,
                'sampled_frames': sampled, 'duration': sampled / args.sample_fps,
                'interval': interval, 'target_height': th, 'target_width': tw,
                'data_size': sampled * th * tw, 'num_chunks': sampled // args.chunk_size
            }
        except Exception as e:
            print(f"  ⚠️ Warning: Failed to read {inp_path}, skipping: {e}")
            return None

    # 3. 多线程并发执行
    video_infos = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_extract_worker, inp, out, ser) for inp, out, ser in tasks]
        
        completed = 0
        for future in as_completed(futures):
            res = future.result()
            if res:
                video_infos.append(res)
            completed += 1
            # 每 1000 个或全部完成时打印进度
            if completed % 1000 == 0 or completed == len(tasks):
                print(f"  Scanned {completed}/{len(tasks)} videos ({completed/len(tasks)*100:.1f}%)")

    if not video_infos:
        print("Error: No valid videos to process after scanning")
        return None
    
    # 4. 排序与分 Batch
    video_infos.sort(key=lambda x: (x.get('series', ''), x['data_size']))
    print(f"  ✅ Total valid videos: {len(video_infos)}")
    return video_infos
    

def run_worker(gpu_id, assigned_videos, args):
    """Independent inference worker for a single GPU."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    decord.bridge.set_bridge("torch")  # Must be set per process
    device = torch.device("cuda:0")
    
    cfg = MODEL_CONFIGS[args.model]
    print(f"[GPU {gpu_id}] Loading model: {cfg['desc']}")
    model = load_model_from_config(cfg["cfg"], cfg["ckpt"])
    model.to(device).eval()

    if hasattr(torch, 'compile'):
        print(f"[GPU {gpu_id}] Compiling model...")
        model.encode = torch.compile(model.encode, mode="reduce-overhead")
        model.decode = torch.compile(model.decode, mode="reduce-overhead")

    time_downsample = model.encoder.time_downsample_factor
    output_token_hz = args.sample_fps / time_downsample

    # Group into local batches
    batches = [assigned_videos[i:i + args.batch_size] for i in range(0, len(assigned_videos), args.batch_size)]
    print(f"[GPU {gpu_id}] Assigned {len(assigned_videos)} videos, organized into {len(batches)} batches.")

    processed_count = 0
    for batch_idx, batch_videos in enumerate(batches):
        all_video_tokens = {}
        all_video_outputs = {}
        print(f"\n{'='*60}")
        print(f"Processing Batch {batch_idx + 1}/{len(batches)} ({len(batch_videos)} videos)")
        print(f"{'='*60}")
        
        tic_batch = time.time()
        for video_idx, video_info in enumerate(batch_videos):
            video_file = video_info['path']
            print(f"[{video_idx + 1}/{len(batch_videos)}] {video_file}")
            dataset = VideoDataset(
                video_path=str(video_file),
                input_height=video_info['target_height'],
                input_width=video_info['target_width'],
                sample_fps=args.sample_fps,
                chunk_size=args.chunk_size,
                is_causal=model.is_causal,
                read_long_video=args.read_long_video,
                num_chunks=video_info['num_chunks'],
            )
            # dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

            video_tokens = []
            video_outputs = []
            video_inputs = []

            with torch.no_grad():
                for chunk_data in dataloader:
                    # chunk = chunk_data[0].to(device)
                    chunk = chunk_data[0].to(device, non_blocking=True)
                    z, reg_log = model.encode(chunk, return_reg_log=True)

                    if 'indices' in reg_log:
                        video_tokens.append(reg_log['indices'])
                    else:
                        video_tokens.append(z)

                    if args.decode:
                        xrec = model.decode(z)
                        video_outputs.append(xrec.clone())
                        video_inputs.append(chunk)

            if len(video_tokens) > 0:
                if 'indices' in reg_log:
                    video_tokens = torch.cat(video_tokens, dim=1)
                else:
                    video_tokens = torch.cat(video_tokens, dim=2)
            else:
                video_tokens = torch.tensor([])
            print(f"  Tokens shape: {video_tokens.shape}")
            all_video_tokens[str(video_file)] = video_tokens

            if args.decode:
                video_outputs = torch.cat(video_outputs, dim=2)
                video_inputs = torch.cat(video_inputs, dim=2)
                if model.is_causal and video_outputs.shape[2] != video_inputs.shape[2]:
                    T_min = min(video_inputs.shape[2], video_outputs.shape[2])
                    video_inputs = video_inputs[:, :, :T_min, :, :]
                    video_outputs = video_outputs[:, :, :T_min, :, :]
                all_video_outputs[str(video_file)] = (video_outputs, video_inputs, video_info)

        # Save results
        for video_info in batch_videos:
            vid_key = str(video_info['path'])
            if vid_key in all_video_tokens:
                save_tokens(all_video_tokens[vid_key].cpu(), video_info['output'], vid_key, output_token_hz)
                processed_count += 1

        toc_batch = time.time()
        batch_duration = sum(v['duration'] for v in batch_videos)
        rtf = (toc_batch - tic_batch) / batch_duration if batch_duration > 0 else 0
        print(f"[GPU {gpu_id}] Batch {batch_idx+1}/{len(batches)} done | RTF: {rtf:.3f} | Total processed: {processed_count}")

        if args.decode:
            for vid_key, (outputs, inputs, video_info) in all_video_outputs.items():
                psnr, ssim = compute_metrics(inputs.cpu(), outputs.cpu())
                print(f"  {video_info['path'].name}: PSNR={psnr:.4f}dB, SSIM={ssim:.4f}")
                out_path = os.path.join(args.output_dir, f"{Path(vid_key).stem}_recon.mp4")
                out_uint8 = tensor_to_uint8(outputs.clamp(-1, 1).squeeze(0))
                write_video(out_path, out_uint8, fps=args.sample_fps)
                print(f"  Saved: {out_path}")
    print(f"[GPU {gpu_id}] ✅ Worker finished. Processed {processed_count} videos.")


def main():
    parser = argparse.ArgumentParser(description="VidTok: Video to Tokens (Multi-GPU)")
    parser.add_argument("--model", type=str, default="kl_noncausal", choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--video", type=str, default=None, help="Input video path or folder")
    parser.add_argument("--series_dir", type=str, default=None, help="Directory containing TV series folders")
    parser.add_argument("--sample_fps", type=int, default=25)
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--chunk_size", type=int, default=16)
    parser.add_argument("--decode", action="store_true")
    parser.add_argument("--read_long_video", action="store_true", default=True)
    parser.add_argument("--adaptive_size", action="store_true", default=True)
    parser.add_argument("--max_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--num_gpus", type=int, default=-1, help="Number of GPUs to use (-1 for all available)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine GPU count
    num_gpus = torch.cuda.device_count() if args.num_gpus == -1 else min(args.num_gpus, torch.cuda.device_count())
    if num_gpus == 0:
        print("Error: No CUDA devices found.")
        sys.exit(1)
    print(f"Detected {num_gpus} GPUs. Launching parallel workers...")

    # Scan videos once in main process
    video_files, series_videos_map = None, None
    if args.video:
        vp = Path(args.video)
        if vp.is_file(): video_files = [vp]
        elif vp.is_dir(): video_files = sorted(list(vp.glob("*.mp4")) + list(vp.glob("*.MP4")))
    elif args.series_dir:
        series_videos_map = get_video_files_by_series(args.series_dir)
        
    all_videos = scan_videos(video_files, series_videos_map, args)
    if not all_videos:
        print("No valid videos to process.")
        return

    # Split evenly among GPUs
    chunks = [[] for _ in range(num_gpus)]
    for i, vid in enumerate(all_videos):
        chunks[i % num_gpus].append(vid)

    # Spawn workers
    processes = []
    for gid in range(num_gpus):
        p = mp.Process(target=run_worker, args=(gid, chunks[gid], args))
        p.start()
        processes.append(p)
        
    for p in processes:
        p.join()
        
    print("\n🎉 All workers completed successfully!")


if __name__ == "__main__":
    main()
