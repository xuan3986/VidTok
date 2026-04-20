#!/usr/bin/env python3
"""
VidTok: Video to Tokens Converter

Usage:
    python infer.py --model kl_noncausal --video assets/example2.mp4 --sample_fps 25 --output_dir ./output
    python infer.py --model kl_noncausal --video ./videos --sample_fps 25 --output_dir ./output --batch_size 8
    
Models: fsq, kl_noncausal, kl_causal, kl_causal_288
-----------------------------------------------------
添加 --decode 以解码视频隐空间特征，特征维度 [Batch, Channels, Time, Height, Width]；
代码会在以下情况下自动从组 batch 降级到顺序处理：
1、batch 内样本分辨率不一致
2、启用 --decode 选项
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
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    },
    "kl_causal_288": {
        "cfg": "configs/vidtok_kl_causal_288_8chn.yaml",
        "ckpt": "checkpoints/vidtok_kl_causal_288_8chn.ckpt",
        "desc": "KL Causal 288 (Continuous, 2x downsampling)",
    },
}


class VideoDataset(Dataset):
    def __init__(
        self,
        video_path: str,
        input_height: int = 256,
        input_width: int = 256,
        sample_fps: int = 25,
        chunk_size: int = 16,
        is_causal: bool = True,
        read_long_video: bool = True,
        adaptive_size: bool = True,
        max_size: int = 256,
    ):
        self.video_path = video_path
        self.video_reader = decord.VideoReader(video_path, num_threads=0)
        self.total_frames = len(self.video_reader)
        self.fps = self.video_reader.get_avg_fps()

        self.original_height, self.original_width = self.video_reader[0].shape[:2]

        if adaptive_size:
            scale = min(max_size / self.original_width, max_size / self.original_height)
            if scale < 1:
                target_width = int(self.original_width * scale)
                target_height = int(self.original_height * scale)
            else:
                target_width = self.original_width
                target_height = self.original_height
            target_width = ((target_width - 1) // 8 + 1) * 8
            target_height = ((target_height - 1) // 8 + 1) * 8
            self.input_height = target_height
            self.input_width = target_width
        else:
            self.input_height = input_height
            self.input_width = input_width

        self.transform = transforms.Compose([
            transforms.Resize((self.input_height, self.input_width), antialias=True),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ])

        interval = round(self.fps / sample_fps)
        self.frame_ids = list(range(0, self.total_frames, interval))

        self.frame_ids_batch = []
        video_length = len(self.frame_ids)

        if read_long_video:
            if is_causal:
                for x in range(0, video_length - 1, chunk_size):
                    end = min(x + chunk_size + 1, video_length)
                    chunk_frame_ids = self.frame_ids[x:end]
                    if len(chunk_frame_ids) == chunk_size + 1:
                        self.frame_ids_batch.append(chunk_frame_ids)
            else:
                for x in range(0, video_length, chunk_size):
                    end = min(x + chunk_size, video_length)
                    chunk_frame_ids = self.frame_ids[x:end]
                    if len(chunk_frame_ids) == chunk_size:
                        self.frame_ids_batch.append(chunk_frame_ids)
        else:
            num_frames_per_batch = chunk_size + 1 if is_causal else chunk_size
            for x in range(0, len(self.frame_ids), num_frames_per_batch):
                chunk_frame_ids = self.frame_ids[x : x + num_frames_per_batch]
                if len(chunk_frame_ids) == num_frames_per_batch:
                    self.frame_ids_batch.append(chunk_frame_ids)

        self.chunks = self.frame_ids_batch
        self.total_sampled_frames = len(self.frame_ids)

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
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

def get_video_files_by_series(series_dir):
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
                # 保持原文件名 stem
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


def build_data_batches(video_files, series_videos_map, args, max_workers=80):
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
                'data_size': sampled * th * tw
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
    print(f"  📏 Data size range: {video_infos[0]['data_size']/1e6:.1f}M - {video_infos[-1]['data_size']/1e6:.1f}M pixels")

    batches = [video_infos[i:i + args.batch_size] for i in range(0, len(video_infos), args.batch_size)]
    print(f"  📦 Organized into {len(batches)} batches (batch_size={args.batch_size})")
    return batches
    

def main():
    parser = argparse.ArgumentParser(description="VidTok: Video to Tokens")
    parser.add_argument("--model", type=str, default="fsq",
                        choices=["fsq", "kl_noncausal", "kl_causal", "kl_causal_288"],
                        help="Model type")
    parser.add_argument("--video", type=str, default=None,
                        help="Input video path or folder containing mp4 files")
    parser.add_argument("--series_dir", type=str, default=None,
                        help="Directory containing TV series folders")
    parser.add_argument("--sample_fps", type=int, default=25,
                        help="Sample FPS for input video")
    parser.add_argument("--output_dir", type=str, default="./output",
                        help="Output directory")
    parser.add_argument("--chunk_size", type=int, default=16,
                        help="Chunk size for long videos")
    parser.add_argument("--decode", action="store_true",
                        help="Decode and save reconstructed video")
    parser.add_argument("--read_long_video", action="store_true", default=True,
                        help="Read long video mode (only process complete chunks)")
    parser.add_argument("--adaptive_size", action="store_true", default=True,
                        help="Adaptively resize video based on original resolution (default: True)")
    parser.add_argument("--max_size", type=int, default=256,
                        help="Maximum resolution for adaptive sizing (default: 256)")
    parser.add_argument("--batch_size", type=int, default=100,
                        help="Batch size for parallel processing")
    args = parser.parse_args()

    cfg = MODEL_CONFIGS[args.model]

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"\nLoading model: {cfg['desc']}")
    model = load_model_from_config(cfg["cfg"], cfg["ckpt"])
    model.to(device).eval()

    if hasattr(torch, 'compile') and device != torch.device("cpu"):
        print("\nCompiling model with Torch Compile...")
        print("(First inference will be slower due to compilation)")
        model.encode = torch.compile(model.encode, mode="reduce-overhead")
        model.decode = torch.compile(model.decode, mode="reduce-overhead")
        print("Model compilation complete!")

    time_downsample = model.encoder.time_downsample_factor
    output_token_hz = args.sample_fps / time_downsample

    print(f"\nModel config:")
    print(f"  Time downsampling: {time_downsample}x")
    print(f"  Input sample_fps: {args.sample_fps}")
    print(f"  Output token rate: {output_token_hz}Hz")
    
    video_files, series_videos_map = None, None
    if args.video:
        video_path = Path(args.video)
        if video_path.is_file():
            video_files = [video_path]
            print(f"\nProcessing single video: {video_path}")
        elif video_path.is_dir():
            video_files = sorted(list(video_path.glob("*.mp4")) + list(video_path.glob("*.MP4")))
            if not video_files:
                print(f"Error: No mp4 files found in {video_path}")
                return
            print(f"\nFound {len(video_files)} videos in {video_path}")
    elif args.series_dir:
        series_videos_map = get_video_files_by_series(args.series_dir)
    else:
        print("Error: Input is None. Please provide --video or --series_dir.")
        sys.exit(1)


    batches = build_data_batches(video_files, series_videos_map, args)
    for batch_idx, batch_videos in enumerate(batches):
        all_video_tokens = {}
        all_video_outputs = {}
        print(f"\n{'='*60}")
        print(f"Processing Batch {batch_idx + 1}/{len(batches)} ({len(batch_videos)} videos)")
        print(f"{'='*60}")

        tic_batch = time.time()
        for video_idx, video_info in enumerate(batch_videos):
            video_file = video_info['path']
            print(f"\n  [{video_idx + 1}/{len(batch_videos)}] {video_file}")

            dataset = VideoDataset(
                video_path=str(video_file),
                sample_fps=args.sample_fps,
                chunk_size=args.chunk_size,
                is_causal=model.is_causal,
                read_long_video=args.read_long_video,
                adaptive_size=args.adaptive_size,
                max_size=args.max_size,
            )
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

            print(f"    Input resolution: {dataset.input_width}x{dataset.input_height} Chunks: {len(dataset.chunks)}")

            video_tokens = []
            video_outputs = []
            video_inputs = []

            with torch.no_grad():
                for chunk_data in dataloader:
                    # chunk_data is a tuple (frames, length)
                    chunk = chunk_data[0]  # extract frames tensor
                    chunk = chunk.to(device)
                    z, reg_log = model.encode(chunk, return_reg_log=True)

                    if 'indices' in reg_log:
                        video_tokens.append(reg_log['indices'].cpu())
                    else:
                        video_tokens.append(z.cpu())

                    if args.decode:
                        xrec = model.decode(z)
                        video_outputs.append(xrec.cpu())
                        video_inputs.append(chunk.cpu())

            # Concatenate tokens along the temporal dimension
            if len(video_tokens) > 0:
                if 'indices' in reg_log:  # FSQ model: [B, T, ...]
                    video_tokens = torch.cat(video_tokens, dim=1)
                else:  # KL model: [B, C, T, H, W]
                    video_tokens = torch.cat(video_tokens, dim=2)
            else:
                video_tokens = torch.tensor([])

            print(f"    Tokens shape: {video_tokens.shape}")
            all_video_tokens[str(video_file)] = video_tokens

            if args.decode:
                video_outputs = torch.cat(video_outputs, dim=2)  # Concatenate along temporal dimension
                video_inputs = torch.cat(video_inputs, dim=2)

                if model.is_causal and video_outputs.shape[2] != video_inputs.shape[2]:
                    T_min = min(video_inputs.shape[2], video_outputs.shape[2])
                    video_inputs = video_inputs[:, :, :T_min, :, :]
                    video_outputs = video_outputs[:, :, :T_min, :, :]

                all_video_outputs[str(video_file)] = (video_outputs, video_inputs, video_info)

        toc_batch = time.time()

        for video_info in batch_videos:
            vid_key = str(video_info['path'])
            if vid_key in all_video_tokens:
                save_tokens(all_video_tokens[vid_key], video_info['output'], vid_key, output_token_hz)
                

        processing_time = toc_batch - tic_batch
        batch_duration = sum(v['duration'] for v in batch_videos)
        batch_rtf = processing_time / batch_duration if batch_duration > 0 else 0

        print(f"\n  Batch {batch_idx + 1} Summary:")
        print(f"    Processing time: {processing_time:.2f}s")
        print(f"    Total duration: {batch_duration:.2f}s")
        print(f"    RTF: {batch_rtf:.4f}")

        if args.decode:
            print(f"\n{'='*60}")
            print("Saving reconstructed videos...")
            for vid_key, (outputs, inputs, video_info) in all_video_outputs.items():
                psnr, ssim = compute_metrics(inputs, outputs)
                print(f"  {video_info['path'].name}: PSNR={psnr:.4f}dB, SSIM={ssim:.4f}")
                video_output_path = os.path.join(args.output_dir, f"{Path(vid_key).stem}_recon.mp4")
                out = outputs.clamp(-1, 1)
                out = out.squeeze(0)  # [C, T, H, W]
                print(f"  Output tensor shape after squeeze: {out.shape}")
                out_uint8 = tensor_to_uint8(out)  # Should be [C, T, H, W]
                write_video(video_output_path, out_uint8, fps=args.sample_fps)
                print(f"  Saved: {video_output_path}")
                

    print(f"\n{'='*60}")
    print(f"All done! Processed {len(all_video_tokens)} videos.")
    print(f"Tokens saved to respective directories.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
