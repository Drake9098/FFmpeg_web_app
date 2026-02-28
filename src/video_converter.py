"""
Video Converter - Main Module
Converts videos to different formats using ffmpeg
"""

import json
import ffmpeg
import subprocess
import time
import os
import re
from pathlib import Path
from typing import Dict, Optional, Tuple
import streamlit as st

class VideoConverter:
    """Main class for video conversion operations"""
    def __init__(self, output_dir: str = "output"):
        """
        Initializer
        
        Args:
            output_dir: Directory where converted videos will be saved
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def get_video_info(self, input_path: str) -> Dict:
        """
        Analyze video properties using ffprobe
        
        Args:
            input_path: Path to input video file
            
        Returns:
            Dictionary containing video information
        """
        try:
            probe = ffmpeg.probe(input_path)
            video_stream = next((stream for stream in probe['streams'] 
                               if stream['codec_type'] == 'video'), None)
            audio_stream = next((stream for stream in probe['streams'] 
                               if stream['codec_type'] == 'audio'), None)
            
            info = {
                'filename': os.path.basename(input_path),
                'format': probe['format']['format_name'],
                'duration': float(probe['format']['duration']),
                'size_bytes': int(probe['format']['size']),
                'size_mb': round(int(probe['format']['size']) / (1024 * 1024), 2),
                'bitrate': int(probe['format']['bit_rate']),
            }
            
            if video_stream:
                info.update({
                    'video_codec': video_stream['codec_name'],
                    'width': int(video_stream['width']),
                    'height': int(video_stream['height']),
                    'fps': eval(video_stream['r_frame_rate']),
                    'video_bitrate': int(video_stream.get('bit_rate', 0)) if 'bit_rate' in video_stream else None,
                })
            
            if audio_stream:
                info.update({
                    'audio_codec': audio_stream['codec_name'],
                    'audio_bitrate': int(audio_stream.get('bit_rate', 0)) if 'bit_rate' in audio_stream else None,
                    'sample_rate': int(audio_stream['sample_rate']),
                })
            
            return info
            
        except ffmpeg.Error as e:
            print(f"Error analyzing video: {e.stderr.decode()}")
            raise

    def convert_video(self,
                     input_path: str,
                     output_format: str = 'mp4',
                     video_codec: str = 'libx264',
                     audio_codec: str = 'aac',
                     crf: int = 23,
                     preset: str = 'medium',
                     resolution: Optional[Tuple[int, int]] = None,
                     video_bitrate: Optional[str] = None,
                     audio_bitrate: Optional[str] = None,
                     progress_bar: Optional[st.progress] = None) -> Dict:
        """
        Convert video to specified format and settings
        
        Args:
            input_path: Path to input video file
            output_format: Output format (mp4, avi, mkv, etc.)
            video_codec: Video codec to use (libx264, libx265, etc.)
            audio_codec: Audio codec to use (aac, mp3, etc.)
            crf: Constant Rate Factor for quality (0-51, lower is better)
            preset: Encoding preset (ultrafast, fast, medium, slow, veryslow)
            resolution: Output resolution as (width, height) tuple
            video_bitrate: Target video bitrate (e.g., '1M', '2000k')
            audio_bitrate: Target audio bitrate (e.g., '128k', '192k')
            
        Returns:
            Dictionary containing conversion statistics
        """
        # Get input video info
        input_info = self.get_video_info(input_path)
        
        # Generate output filename
        input_name = Path(input_path).stem
        output_path = self.output_dir / f"{input_name}_converted.{output_format}"
        
        print(f"\n{'='*60}")
        print(f"Converting: {input_info['filename']}")
        print(f"Original: {input_info['width']}x{input_info['height']}, "
              f"{input_info['size_mb']} MB, {input_info['video_codec']}")
        print(f"{'='*60}\n")
        
        # Build ffmpeg command
        input_stream = ffmpeg.input(input_path)
        video_stream = input_stream.video
        
        # Video encoding options
        video_options = {
            'c:v': video_codec,
            'preset': preset,
        }
        
        if crf is not None and video_bitrate is None:
            video_options['crf'] = crf
        
        if video_bitrate:
            video_options['b:v'] = video_bitrate
        
        # Audio encoding options
        audio_options = {
            'c:a': audio_codec,
        }
        
        if audio_bitrate:
            audio_options['b:a'] = audio_bitrate
        
        # Apply resolution scaling if specified
        if resolution:
            video_stream = ffmpeg.filter(video_stream, 'scale', resolution[0], resolution[1])
        
        # Prepare output streams
        output_streams = [video_stream]
        if 'audio_codec' in input_info:
            output_streams.append(input_stream.audio)
        
        # Combine options
        output_options = {**video_options, **audio_options}
        
        # Start conversion
        start_time = time.time()
        
        try:
            stream = ffmpeg.output(*output_streams, str(output_path), **output_options)
            cmd = ffmpeg.compile(stream, overwrite_output=True)
            
            
            duration = input_info['duration']
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            
            # Read output in real-time to show progress in Streamlit
            while True:
                output = process.stderr.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    # Parse progress information from ffmpeg output
                    time_match = re.search(r'time=(\d+:\d+:\d+\.\d+)', output)
                    if time_match and progress_bar is not None:
                        elapsed_time_str = time_match.group(1)
                        h, m, s = map(float, elapsed_time_str.split(':'))
                        elapsed_seconds = h * 3600 + m * 60 + s
                        progress_percent = min(int((elapsed_seconds / duration) * 100), 100)
                        progress_bar.progress(progress_percent)
            

            process.wait()
            if process.returncode != 0:
                raise Exception("FFmpeg conversion failed")
                        
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Get output video info
            output_info = self.get_video_info(str(output_path))
            
            # Calculate statistics
            space_saved_mb = input_info['size_mb'] - output_info['size_mb']
            space_saved_percent = (space_saved_mb / input_info['size_mb']) * 100
            compression_ratio = input_info['size_mb'] / output_info['size_mb']
            
            stats = {
                'input_file': input_info['filename'],
                'output_file': output_path.name,
                'input_size_mb': input_info['size_mb'],
                'output_size_mb': output_info['size_mb'],
                'space_saved_mb': round(space_saved_mb, 2),
                'space_saved_percent': round(space_saved_percent, 2),
                'compression_ratio': round(compression_ratio, 2),
                'execution_time_seconds': round(execution_time, 2),
                'input_resolution': f"{input_info['width']}x{input_info['height']}",
                'output_resolution': f"{output_info['width']}x{output_info['height']}",
                'input_codec': input_info['video_codec'],
                'output_codec': output_info['video_codec'],
            }
            
            # Print results to console
            print(f"\n{'='*60}")
            print(f"Conversion Complete!")
            print(f"{'='*60}")
            print(f"Output file: {output_path.name}")
            print(f"Execution time: {stats['execution_time_seconds']} seconds")
            print(f"Original size: {stats['input_size_mb']} MB")
            print(f"New size: {stats['output_size_mb']} MB")
            print(f"Space saved: {stats['space_saved_mb']} MB ({stats['space_saved_percent']}%)")
            print(f"Compression ratio: {stats['compression_ratio']}x")
            print(f"{'='*60}\n")

            if 'ssim_mean' in stats:
                print(f"SSIM: {stats['ssim_mean']:.4f} (Quality: {stats['quality_assessment']})")
                print(f"PSNR: {stats['psnr_mean']:.2f} dB (Higher is better)")
                print(f"{'='*60}\n")
            
            return stats
            
        except ffmpeg.Error as e:
            print(f"Error during conversion: {e.stderr.decode()}")
            raise
