import re
import sys
import time
import json
import pysrt
import ffmpeg
import logging
import traceback
import configparser
import multiprocessing
from queue import Empty
from pathlib import Path
from datetime import timedelta
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QFileDialog, QLineEdit, QGroupBox,
                             QFormLayout, QSpinBox, QTextEdit, QComboBox,
                             QProgressBar, QMessageBox, QCheckBox)
from PyQt5.QtCore import QObject, QThread, pyqtSignal, QUrl
from PyQt5.QtGui import QDesktopServices, QIcon
import dashscope
from dashscope import Generation
try:
    import resources
except ImportError:
    resources = None

class Config:
    APP_NAME = "颜趣AI高光剪辑工具"
    APP_VERSION = "1.2.0"

STYLESHEET = """
QWidget { background-color: #1A1D2A; color: #D0D0D0; font-family: "Segoe UI", "Microsoft YaHei", "PingFang SC", sans-serif; font-size: 10pt; }
QGroupBox { background-color: #24283B; border: 1px solid #25F4EE; border-radius: 8px; margin-top: 1.2em; padding: 15px; }
QGroupBox::title { color: #25F4EE; font-weight: bold; font-size: 11pt; subcontrol-origin: margin; subcontrol-position: top center; padding: 4px 12px; background-color: #1A1D2A; border-radius: 6px; border: 1px solid #25F4EE; }
QPushButton { background-color: #3B3F51; color: #FFFFFF; border: 1px solid #25F4EE; padding: 6px 15px; font-weight: bold; border-radius: 6px; }
QPushButton:hover { background-color: #4A4E60; border-color: #97FEFA; }
QPushButton:pressed { background-color: #25F4EE; color: #1A1D2A; }
QPushButton:disabled { background-color: #4A4E60; color: #888888; border-color: #555555; }
QPushButton#PrimaryButton { background-color: #FE2C55; border-color: #FE2C55; color: #ffffff; font-size: 12pt; font-weight: bold; }
QPushButton#PrimaryButton:hover { background-color: #FF4D71; border-color: #FF4D71; }
QPushButton#PrimaryButton:pressed { background-color: #D92349; }
QTextEdit, QLineEdit { background-color: #1A1D2A; color: #FFFFFF; border: 1px solid #4A4E60; padding: 6px; border-radius: 6px; selection-background-color: #25F4EE; selection-color: #1A1D2A; }
QTextEdit:focus, QLineEdit:focus { border-color: #25F4EE; }
QLabel { color: #E0E0E0; background-color: transparent; }
QSpinBox, QComboBox { background-color: #3B3F51; border: 1px solid #25F4EE; border-radius: 5px; padding: 5px; color: #FFFFFF; min-width: 6em; }
QComboBox::drop-down { subcontrol-origin: padding; subcontrol-position: top right; width: 20px; border-left-width: 1px; border-left-color: #4A4E60; border-left-style: solid; border-top-right-radius: 5px; border-bottom-right-radius: 5px; }
QComboBox::down-arrow { image: url(:/down_arrow.png); width: 12px; height: 12px; }
QComboBox QAbstractItemView { background-color: #3B3F51; color: #FFFFFF; selection-background-color: #25F4EE; selection-color: #1A1D2A; border: 1px solid #4A4E60; }
QProgressBar { border: 1px solid #4A4E60; border-radius: 5px; text-align: center; color: #FFFFFF; background-color: #3B3F51; height: 10px; }
QProgressBar::chunk { background-color: #25F4EE; border-radius: 5px; }
QScrollBar:vertical, QScrollBar:horizontal { border: none; background-color: #24283B; width: 10px; margin: 0px; }
QScrollBar::handle:vertical, QScrollBar::handle:horizontal { background-color: #4A4E60; border-radius: 5px; min-height: 20px; min-width: 20px; }
QScrollBar::handle:vertical:hover, QScrollBar::handle:horizontal:hover { background-color: #25F4EE; }
QScrollBar::add-line, QScrollBar::sub-line { border: none; background: none; height: 0; width: 0; }
QScrollBar::up-arrow:vertical, QScrollBar::down-arrow:vertical, QScrollBar::left-arrow:horizontal, QScrollBar::right-arrow:horizontal { background: none; }
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical, QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal { background: none; }
QMessageBox { background-color: #24283B; }
QMessageBox QLabel { color: #FFFFFF; background-color: transparent; }
QMenuBar { background-color: #1A1D2A; border-bottom: 1px solid #25F4EE; }
QMenuBar::item { background-color: transparent; color: #D0D0D0; padding: 6px 12px; }
QMenuBar::item:selected, QMenuBar::item:hover { background-color: #3B3F51; }
QMenu { background-color: #24283B; border: 1px solid #4A4E60; padding: 5px; }
QMenu::item { color: #D0D0D0; padding: 8px 25px; }
QMenu::item:selected { background-color: #FE2C55; }
QStatusBar { background-color: #1A1D2A; border-top: 1px solid #25F4EE; }
QStatusBar::item { border: none; }
QCheckBox { spacing: 5px; }
QCheckBox::indicator { width: 18px; height: 18px; border-radius: 4px; }
QCheckBox::indicator:unchecked { background-color: #3B3F51; border: 1px solid #4A4E60; }
QCheckBox::indicator:checked { background-color: #25F4EE; border: 1px solid #25F4EE; image: url(:/check.png); }
"""

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
if not logger.handlers:
    logger.addHandler(logging.StreamHandler())

class Clip(BaseModel):
    id: str
    outline: Dict[str, Any]
    content: List[str]
    start_time: str
    end_time: str
    chunk_index: int
    final_score: float = 0.0
    recommend_reason: str = ""
    generated_title: str = ""
    keyword_density: float = 0.0

class LLMClient:
    def __init__(self, config):
        self.provider = config.get('api', 'provider')
        self.api_key = config.get('api', 'api_key')
        self.model_name = config.get('api', 'model_name')
        if not self.api_key or self.api_key == 'sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx' or '...' in self.api_key:
            raise ValueError("请在GUI界面或 config.ini 文件中配置您的有效 API 密钥。")
        dashscope.api_key = self.api_key
    def call(self, prompt: str, input_data: Any) -> Optional[str]:
        full_prompt = f"{prompt}\n\n{json.dumps(input_data, ensure_ascii=False, indent=2)}"
        try:
            response = Generation.call(model=self.model_name, prompt=full_prompt)
            if response.status_code == 200:
                return response.output.text
            else:
                logger.error(f"API 调用失败: {response.code} - {response.message}")
                return None
        except Exception as e:
            logger.error(f"API 调用时发生异常: {e}")
            return None
    def call_with_retry(self, prompt: str, input_data: Any, max_retries: int = 3) -> Optional[str]:
        for attempt in range(max_retries):
            result = self.call(prompt, input_data)
            if result:
                return result
            logger.warning(f"第 {attempt + 1} 次 API 调用返回空或失败，正在重试...")
            time.sleep(2 ** attempt)
        logger.error(f"API 调用在 {max_retries} 次重试后彻底失败。")
        return None
    def parse_json_response(self, response: str) -> Optional[Any]:
        cleaned_response = re.sub(r'```json\s*([\s\S]*?)\s*```', r'\1', response.strip())
        try:
            return json.loads(cleaned_response)
        except json.JSONDecodeError:
            logger.error(f"无法解析 LLM 返回的 JSON: {cleaned_response[:200]}...")
            return None

class SubtitleService:
    def __init__(self, log_callback=logging.info):
        self.log_callback = log_callback
    def optimize_line_breaks(self, input_srt_path: str, output_srt_path: str, max_chars_per_line: int = 12, max_lines_per_sub: int = 1) -> bool:
        self.log_callback(f"  🔄 正在使用高级算法优化字幕: {Path(input_srt_path).name}")
        if not Path(input_srt_path).exists():
            self.log_callback(f"  🔴 错误: 字幕文件 '{input_srt_path}' 不存在。")
            return False
        try:
            subs = pysrt.open(input_srt_path, encoding='utf-8')
            char_timestamps = []
            full_text_list = []
            for sub in subs:
                text = sub.text_without_tags.strip().replace('\n', ' ')
                if not text: continue
                duration_ms = sub.end.ordinal - sub.start.ordinal
                time_per_char = duration_ms / len(text) if len(text) > 0 else 0
                current_time_ms = sub.start.ordinal
                for char in text:
                    char_timestamps.append(pysrt.SubRipTime.from_ordinal(int(current_time_ms)))
                    current_time_ms += time_per_char
                full_text_list.append(text)
            full_text = "".join(full_text_list)
            if not full_text:
                self.log_callback("  ⚠️ 警告: 字幕文件内容为空。")
                with open(output_srt_path, 'w', encoding='utf-8') as f:
                    pass
                return True
            clauses = re.split(r'([，。！？、,.:;!?])', full_text)
            semantic_clauses = [clauses[i] + (clauses[i+1] if i + 1 < len(clauses) and clauses[i+1] in '，。！？、,.:;!?' else '') for i in range(0, len(clauses), 2) if clauses[i]]
            final_lines = []
            for clause in semantic_clauses:
                clause = clause.strip()
                if not clause: continue
                while len(clause) > max_chars_per_line:
                    final_lines.append(clause[:max_chars_per_line])
                    clause = clause[max_chars_per_line:]
                if clause:
                    final_lines.append(clause)
            new_subs = pysrt.SubRipFile()
            char_offset = 0
            current_sub_lines = []
            for line in final_lines:
                if len(current_sub_lines) < max_lines_per_sub:
                    current_sub_lines.append(line)
                else:
                    text_block_lines = current_sub_lines
                    new_text = '\n'.join(text_block_lines).strip()
                    text_length_for_sub = len("".join(text_block_lines))
                    if new_text and char_offset + text_length_for_sub <= len(char_timestamps):
                        start_char_idx = char_offset
                        end_char_idx = char_offset + text_length_for_sub - 1
                        start_time = char_timestamps[start_char_idx]
                        end_time = char_timestamps[end_char_idx]
                        if end_time < start_time:
                            end_time = start_time + timedelta(milliseconds=200 * len(new_text))
                        new_subs.append(pysrt.SubRipItem(index=len(new_subs) + 1, start=start_time, end=end_time, text=new_text))
                    char_offset += text_length_for_sub
                    current_sub_lines = [line]
            if current_sub_lines:
                text_block_lines = current_sub_lines
                new_text = '\n'.join(text_block_lines).strip()
                text_length_for_sub = len("".join(text_block_lines))
                if new_text and char_offset + text_length_for_sub <= len(char_timestamps):
                    start_char_idx = char_offset
                    end_char_idx = char_offset + text_length_for_sub - 1
                    start_time = char_timestamps[start_char_idx]
                    end_time = char_timestamps[end_char_idx]
                    if end_time < start_time:
                        end_time = start_time + timedelta(milliseconds=200 * len(new_text))
                    new_subs.append(pysrt.SubRipItem(index=len(new_subs) + 1, start=start_time, end=end_time, text=new_text))
            new_subs.save(output_srt_path, encoding='utf-8')
            self.log_callback(f"  ✅ 字幕优化完成 -> {Path(output_srt_path).name}")
            return True
        except Exception as e:
            self.log_callback(f"  🔴 处理字幕时发生严重错误: {e}\n{traceback.format_exc()}")
            return False

class Processor:
    @staticmethod
    def time_to_seconds(time_str: str) -> float:
        try:
            parts = time_str.replace(',', '.').split(':')
            h, m = int(parts[0]), int(parts[1])
            s_ms_parts = parts[2].split('.')
            s, ms = int(s_ms_parts[0]), int(s_ms_parts[1])
            return h * 3600 + m * 60 + s + ms / 1000
        except (IndexError, ValueError) as e:
            logger.error(f"解析时间字符串 '{time_str}'失败: {e}")
            return 0.0
    @staticmethod
    def seconds_to_time_str(seconds: float) -> str:
        h, m, s = int(seconds // 3600), int((seconds % 3600) // 60), int(seconds % 60)
        ms = int((seconds - int(seconds)) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
    @staticmethod
    def get_video_duration(video_path: Path) -> float:
        try:
            return float(ffmpeg.probe(str(video_path))['format']['duration'])
        except (ffmpeg.Error, Exception) as e:
            logger.error(f"无法获取视频 '{video_path.name}' 的时长: {getattr(e, 'stderr', e)}")
            return 0.0
    @staticmethod
    def parse_srt(srt_path: Path) -> List[Dict]:
        with open(srt_path, 'r', encoding='utf-8') as f:
            content = f.read()
        entries = []
        for block in content.strip().split('\n\n'):
            lines = block.split('\n')
            if len(lines) >= 3:
                try:
                    time_match = re.match(r'(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})', lines[1])
                    if time_match:
                        start, end = time_match.groups()
                        text = ' '.join(lines[2:])
                        entries.append({"index": int(lines[0]), "start_time": start, "end_time": end, "text": text})
                except (ValueError, IndexError):
                    continue
        return entries
    @staticmethod
    def cut_video_clip(source_path: Path, output_path: Path, start_seconds: float, end_seconds: float, subtitle_path: Optional[Path] = None, font_file: Optional[str] = None):
        try:
            if start_seconds < 0 or end_seconds <= start_seconds:
                logger.error(f"跳过剪辑，无效的时间戳: start={start_seconds:.2f}, end={end_seconds:.2f} for {output_path.name}")
                return
            duration = end_seconds - start_seconds
            ffmpeg_executable = "ffmpeg.exe" if sys.platform == "win32" else "ffmpeg"
            stream = ffmpeg.input(str(source_path), ss=start_seconds)
            video_stream = stream.video
            audio_stream = stream.audio
            video_stream = ffmpeg.filter(video_stream, 'scale', '1920', '-2')
            if subtitle_path and subtitle_path.exists() and subtitle_path.stat().st_size > 0 and font_file:
                font_dir = (Path.cwd() / "fonts").as_posix()
                style = f"Fontname={font_file},PrimaryColour=&H00FFFF&,OutlineColour=&HFF000000,BorderStyle=1,Outline=2,Shadow=1,MarginV=5,Alignment=2"
                video_stream = ffmpeg.filter(video_stream, 'subtitles', filename=subtitle_path.as_posix(), force_style=style, fontsdir=font_dir)
                logger.info(f"正在为 {output_path.name} 添加字幕: {subtitle_path.name}")
            (
                ffmpeg.output(
                    video_stream,
                    audio_stream,
                    str(output_path),
                    t=duration,
                    vcodec='libx264',
                    acodec='aac',
                    preset='fast',
                    crf=23,
                    strict='experimental'
                )
                .run(cmd=ffmpeg_executable, overwrite_output=True, quiet=True)
            )
            logger.info(f"成功生成片段: {output_path.name} (时长: {duration:.2f}s)")
        except ffmpeg.Error as e:
            logger.error(f"FFmpeg 剪辑失败: {output_path.name}\n{e.stderr.decode('utf-8', 'replace') if e.stderr else 'N/A'}")
        except Exception as e:
            logger.error(f"剪辑视频时发生未知错误: {e}")
    @staticmethod
    def generate_sliding_window_clips(srt_data: List[Dict], target_duration: int, tolerance: float, slide_step: int) -> List[Dict]:
        clips = []
        min_duration = target_duration * (1 - tolerance)
        max_duration = target_duration * (1 + tolerance)
        if not srt_data:
            return []
        next_start_time = 0.0
        for i in range(len(srt_data)):
            start_entry_time = Processor.time_to_seconds(srt_data[i]['start_time'])
            if start_entry_time < next_start_time:
                continue
            current_window_start_time = start_entry_time
            current_content = []
            for j in range(i, len(srt_data)):
                if (Processor.time_to_seconds(srt_data[j]['end_time']) - current_window_start_time) < max_duration:
                    current_content.append(srt_data[j])
                else:
                    current_content.append(srt_data[j])
                    break
            if not current_content:
                continue
            if (Processor.time_to_seconds(current_content[-1]['end_time']) - Processor.time_to_seconds(current_content[0]['start_time'])) >= min_duration:
                clips.append({
                    "outline": {"title": " ".join([e['text'] for e in current_content])[:30] + "...", "subtopics": []},
                    "content": [e['text'] for e in current_content],
                    "start_time": current_content[0]['start_time'],
                    "end_time": current_content[-1]['end_time']
                })
                next_start_time = current_window_start_time + slide_step
        return clips

class AIPipeline:
    def __init__(self, config: configparser.ConfigParser, temp_dir: Path, clip_params: Dict, progress_signal = None):
        self.clip_params = clip_params
        self.llm = LLMClient(config)
        self.processor = Processor()
        self.temp_dir = temp_dir
        self.max_concurrent_tasks = config.getint('settings', 'max_concurrent_tasks', fallback=10)
        self.prompts = {}
        prompt_dir = Path(__file__).parent / "prompts"
        for key in ["推荐理由", "标题生成"]:
            prompt_file = prompt_dir / f"{key}.txt"
            if not prompt_file.exists():
                raise FileNotFoundError(f"提示词文件未找到: {prompt_file}")
            with open(prompt_file, 'r', encoding='utf-8') as f:
                self.prompts[key] = f.read()
        self.progress_callback = progress_signal
    def _emit_progress(self, start_percent: int, current_step: int, total_steps: int, step_weight: int):
        if self.progress_callback and total_steps > 0:
            self.progress_callback(start_percent + int((current_step / total_steps) * step_weight))
    def _batch_list(self, data: List[Any], batch_size: int) -> List[List[Any]]:
        return [data[i:i + batch_size] for i in range(0, len(data), batch_size)]
    def run(self, srt_path: Path) -> List[Clip]:
        srt_data = self.processor.parse_srt(srt_path)
        target_duration, tolerance = self.clip_params['clip_duration'], self.clip_params['duration_tolerance']
        slide_step = max(1, int(target_duration / 2))
        logger.info(f"--- 正在使用滑动窗口生成候选片段 (时长: {target_duration}s, 步长: {slide_step}s, 容忍度: {tolerance:.2f}) ---")
        candidate_clips_data = self.processor.generate_sliding_window_clips(srt_data, target_duration, tolerance, slide_step)
        if not candidate_clips_data:
            logger.warning("未能生成任何符合时长要求的候选片段。")
            return []
        logger.info(f"生成了 {len(candidate_clips_data)} 个候选片段，准备进行AI评分。")
        scored_outlines = self._score_clips(candidate_clips_data) or candidate_clips_data
        logger.info("完成内容评分。")
        all_clips = [Clip(id=str(i), chunk_index=0, **item) for i, item in enumerate(scored_outlines)]
        for clip in all_clips:
            if 'final_score' not in clip.model_dump():
                clip.final_score = 0.0
            if 'recommend_reason' not in clip.model_dump():
                clip.recommend_reason = "N/A"
        if not all_clips:
            return []
        logger.info("--- 正在为所有候选片段生成标题 ---")
        clips_with_titles = self._generate_titles(all_clips)
        logger.info("标题生成完毕。")
        return clips_with_titles
    def _score_clips(self, clips: List[Dict]) -> List[Dict]:
        clip_batches = self._batch_list(clips, 20)
        all_scored_clips = []
        total_batches = len(clip_batches)
        logger.info(f"片段将被分为 {total_batches} 个批次进行并发评分 (最多 {self.max_concurrent_tasks} 个并发任务)。")
        start_progress, weight, completed_batches = 30, 35, 0
        with ThreadPoolExecutor(max_workers=self.max_concurrent_tasks) as executor:
            future_to_index = {executor.submit(self.llm.call_with_retry, self.prompts['推荐理由'], batch): i for i, batch in enumerate(clip_batches)}
            for future in as_completed(future_to_index):
                batch_index = future_to_index[future]
                completed_batches += 1
                try:
                    response = future.result()
                    if response and (parsed_json := self.llm.parse_json_response(response)) and isinstance(parsed_json, list):
                        all_scored_clips.extend(parsed_json)
                        logger.info(f"批次 {batch_index + 1}/{total_batches} 评分成功。")
                    else:
                        logger.warning(f"无法解析或API调用失败，批次 {batch_index + 1} 的评分响应。")
                except Exception as exc:
                    logger.error(f"处理评分批次 {batch_index + 1} 时发生异常: {exc}")
                finally:
                    self._emit_progress(start_progress, completed_batches, total_batches, weight)
        self._emit_progress(start_progress, total_batches, total_batches, weight)
        return all_scored_clips
    def _generate_titles(self, clips: List[Clip]) -> List[Clip]:
        if not clips:
            return []
        clip_batches = self._batch_list(clips, 20)
        clips_by_id = {c.id: c for c in clips}
        total_batches = len(clip_batches)
        logger.info(f"片段将被分为 {total_batches} 个批次并发生成标题。")
        start_progress, weight, completed_batches = 65, 15, 0
        with ThreadPoolExecutor(max_workers=self.max_concurrent_tasks) as executor:
            batch_inputs = [[{"id": c.id, "title": c.outline.get('title', ''), "content": c.content, "recommend_reason": c.recommend_reason} for c in batch] for batch in clip_batches]
            future_to_index = {executor.submit(self.llm.call_with_retry, self.prompts['标题生成'], batch_input): i for i, batch_input in enumerate(batch_inputs)}
            for future in as_completed(future_to_index):
                batch_index = future_to_index[future]
                completed_batches += 1
                try:
                    response = future.result()
                    if response and (titles_map := self.llm.parse_json_response(response)) and isinstance(titles_map, dict):
                        for clip_id, title in titles_map.items():
                            if clip_id in clips_by_id:
                                clips_by_id[clip_id].generated_title = title
                        logger.info(f"批次 {batch_index + 1}/{total_batches} 标题生成成功。")
                    else:
                        logger.warning(f"无法解析或API调用失败，批次 {batch_index + 1} 的标题响应。")
                except Exception as exc:
                    logger.error(f"处理标题生成批次 {batch_index + 1} 时发生异常: {exc}")
                finally:
                    self._emit_progress(start_progress, completed_batches, total_batches, weight)
        self._emit_progress(start_progress, total_batches, total_batches, weight)
        return list(clips_by_id.values())

class QueueHandler(logging.Handler):
    def __init__(self, queue):
        super().__init__()
        self.queue = queue
    def emit(self, record):
        self.queue.put(('log', self.format(record)))

class PyQtLogHandler(logging.Handler):
    def __init__(self, log_signal):
        super().__init__()
        self.log_signal = log_signal
    def emit(self, record):
        self.log_signal.emit(self.format(record))

def run_processing_task(queue, params):
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()
    queue_handler = QueueHandler(queue)
    queue_handler.setFormatter(formatter)
    root_logger.addHandler(queue_handler)
    global logger
    logger = logging.getLogger('worker_process')
    def progress_callback(value):
        queue.put(('progress', value))
    try:
        config = configparser.ConfigParser()
        config.read(params['config_path'], encoding='utf-8')
        input_path = Path(params['input_path'])
        output_dir = Path(params['output_dir'])
        output_dir.mkdir(exist_ok=True, parents=True)
        temp_dir = output_dir / "temp"
        temp_dir.mkdir(exist_ok=True)
        progress_callback(5)
        combined_srt_name = f"{input_path.stem if input_path.is_file() else input_path.name}_combined.srt"
        srt_path = temp_dir / combined_srt_name
        video_files = _get_video_files(input_path)
        if not video_files:
            raise FileNotFoundError("在指定路径下未找到有效的视频文件。")
        _generate_and_combine_srts(video_files, srt_path, temp_dir, params, progress_callback)
        progress_callback(30)
        pipeline = AIPipeline(config, temp_dir, params, progress_signal=progress_callback)
        all_potential_clips = pipeline.run(srt_path)
        if not all_potential_clips:
            raise RuntimeError("AI分析未能生成任何候选片段。")
        logger.info(f"AI分析完成，共生成 {len(all_potential_clips)} 个候选片段。")
        num_clips = params['num_clips']
        logger.info(f"开始筛选片段：数量={num_clips}")
        power_words = params.get('power_words', [])
        def is_overlapping(clip1, clip2, threshold=0.5):
            start1, end1 = Processor.time_to_seconds(clip1.start_time), Processor.time_to_seconds(clip1.end_time)
            start2, end2 = Processor.time_to_seconds(clip2.start_time), Processor.time_to_seconds(clip2.end_time)
            overlap_start = max(start1, start2)
            overlap_end = min(end1, end2)
            if overlap_start < overlap_end:
                overlap_duration = overlap_end - overlap_start
                min_duration = min(end1 - start1, end2 - start2)
                if min_duration > 0 and (overlap_duration / min_duration) > threshold:
                    return True
            return False
        use_keyword_sorting = bool(power_words)
        if use_keyword_sorting:
            logger.info(f"使用自定义关键词进行辅助排序: {', '.join(power_words)}")
            for clip in all_potential_clips:
                content_text = "".join(clip.content)
                keyword_count = sum(content_text.count(word) for word in power_words)
                clip.keyword_density = keyword_count / len(content_text) if len(content_text) > 0 else 0
        else:
            logger.info("未提供高光关键词，仅按分数排序。")
            for clip in all_potential_clips:
                clip.keyword_density = 0
        all_potential_clips.sort(key=lambda c: (c.final_score, c.keyword_density), reverse=True)
        final_clips = []
        for clip in all_potential_clips:
            if len(final_clips) >= num_clips: break
            if not any(is_overlapping(clip, selected_clip) for selected_clip in final_clips):
                final_clips.append(clip)
        logger.info(f"已通过智能筛选和去重，选出 {len(final_clips)} 个最佳片段准备剪辑。")
        burn_subtitles = params.get('burn_subtitles', False)
        font_file_name = None
        if burn_subtitles:
            font_dir = Path.cwd() / "fonts"
            if font_dir.is_dir():
                font_files = list(font_dir.glob('*.ttf')) + list(font_dir.glob('*.otf'))
                if font_files:
                    font_file_name = font_files[0].name
                    logger.info(f"找到并设置字体: {font_file_name}")
                else:
                    logger.warning("勾选了添加字幕,但 'fonts' 文件夹为空。将不添加字幕。")
                    burn_subtitles = False
            else:
                logger.warning("勾选了添加字幕,但 'fonts' 文件夹不存在。将不添加字幕。")
                burn_subtitles = False
        srt_data = Processor.parse_srt(srt_path) if burn_subtitles else []
        progress_callback(80)
        for i, clip in enumerate(final_clips):
            start_s, end_s = Processor.time_to_seconds(clip.start_time), Processor.time_to_seconds(clip.end_time)
            current_offset, source_file_found = 0.0, False
            for video_file in video_files:
                video_duration = Processor.get_video_duration(video_file)
                if start_s < current_offset + video_duration:
                    clip_start_in_file, clip_end_in_file = start_s - current_offset, end_s - current_offset
                    if clip_start_in_file < 0 or clip_end_in_file <= clip_start_in_file:
                        logger.error(f"跳过片段，时间戳无效: start={clip_start_in_file:.2f}, end={clip_end_in_file:.2f}")
                        break
                    safe_title = re.sub(r'[\\/*?:"<>|]', "", clip.generated_title)
                    output_filename = f"{i+1:02d}_{safe_title[:50]}.mp4"
                    clip_subtitle_path = None
                    if burn_subtitles and srt_data and font_file_name:
                        clip_srt_content, sub_index = "", 1
                        for entry in srt_data:
                            entry_start_s, entry_end_s = Processor.time_to_seconds(entry['start_time']), Processor.time_to_seconds(entry['end_time'])
                            if max(start_s, entry_start_s) < min(end_s, entry_end_s):
                                new_start_s, new_end_s = max(0, entry_start_s - start_s), max(0, entry_end_s - start_s)
                                if new_end_s > new_start_s:
                                    clip_srt_content += f"{sub_index}\n{Processor.seconds_to_time_str(new_start_s)} --> {Processor.seconds_to_time_str(new_end_s)}\n{entry['text']}\n\n"
                                    sub_index += 1
                        if clip_srt_content:
                            raw_clip_srt_path = temp_dir / f"raw_clip_{i+1:02d}.srt"
                            with open(raw_clip_srt_path, 'w', encoding='utf-8') as f: f.write(clip_srt_content)
                            if raw_clip_srt_path.stat().st_size > 0:
                                subtitle_optimizer = SubtitleService(log_callback=logger.info)
                                optimized_clip_srt_path = temp_dir / f"optimized_clip_{i+1:02d}.srt"
                                clip_subtitle_path = optimized_clip_srt_path if subtitle_optimizer.optimize_line_breaks(str(raw_clip_srt_path), str(optimized_clip_srt_path)) else raw_clip_srt_path
                    Processor.cut_video_clip(video_file, output_dir / output_filename, clip_start_in_file, clip_end_in_file, subtitle_path=clip_subtitle_path, font_file=font_file_name)
                    source_file_found = True
                    break
                else:
                    current_offset += video_duration
            if not source_file_found:
                logger.warning(f"未能为片段 '{clip.generated_title}' 找到源视频文件。")
            progress_callback(80 + int(20 * (i + 1) / len(final_clips)))
        logger.info(f"🎉 所有高光片段已生成！文件保存在目录: {output_dir.resolve()}")
    except Exception as e:
        logger.error(f"处理过程中发生错误: {e}", exc_info=True)
        queue.put(('error', str(e)))
    finally:
        queue.put(('finished', None))

def _get_video_files(input_path: Path) -> List[Path]:
    VIDEO_EXTENSIONS = ['.mp4', '.mkv', '.avi', '.mov', '.flv', '.wmv']
    if input_path.is_dir():
        logger.info(f"输入为目录，正在扫描视频文件: {input_path}")
        files = sorted([p for p in input_path.iterdir() if p.suffix.lower() in VIDEO_EXTENSIONS])
        logger.info(f"发现 {len(files)} 个视频文件。")
        return files
    elif input_path.is_file() and input_path.suffix.lower() in VIDEO_EXTENSIONS:
        logger.info(f"输入为单个视频文件: {input_path}")
        return [input_path]
    return []

def _generate_and_combine_srts(video_files: List[Path], combined_srt_path: Path, temp_dir: Path, params: dict, progress_callback):
    import whisper
    language_to_use = params.get('language')
    whisper_model_name = params.get('whisper_model', 'medium')
    logger.info(f"正在加载 Whisper 模型: {whisper_model_name}...")
    whisper_model = whisper.load_model(whisper_model_name)
    logger.info("Whisper 模型加载完毕。")
    total_offset_seconds, combined_srt_content, entry_index = 0.0, "", 1
    transcribe_progress_total_weight = 25
    for i, video_file in enumerate(video_files):
        logger.info(f"--- 开始处理视频 {i+1}/{len(video_files)}: {video_file.name} ---")
        individual_srt_path = temp_dir / f"{video_file.stem}.srt"
        if individual_srt_path.exists():
            logger.info(f"发现已存在的SRT文件: '{individual_srt_path.name}'，将直接使用。")
            result = {'segments': [{'start': Processor.time_to_seconds(e['start_time']), 'end': Processor.time_to_seconds(e['end_time']), 'text': e['text']} for e in Processor.parse_srt(individual_srt_path)]}
        else:
            log_msg = f"正在进行语音识别 (语言: {language_to_use or '自动检测'})..."
            logger.info(log_msg)
            transcribe_args = {'language': language_to_use} if language_to_use else {}
            result = whisper_model.transcribe(str(video_file), verbose=False, **transcribe_args)
            individual_srt_content = "".join(f"{seg_idx+1}\n{Processor.seconds_to_time_str(segment['start'])} --> {Processor.seconds_to_time_str(segment['end'])}\n{segment['text'].strip()}\n\n" for seg_idx, segment in enumerate(result['segments']))
            with open(individual_srt_path, 'w', encoding='utf-8') as f:
                f.write(individual_srt_content)
            logger.info(f"已为 '{video_file.name}' 生成独立的SRT文件: {individual_srt_path.name}")
        for segment in result['segments']:
            start_str = Processor.seconds_to_time_str(segment['start'] + total_offset_seconds)
            end_str = Processor.seconds_to_time_str(segment['end'] + total_offset_seconds)
            combined_srt_content += f"{entry_index}\n{start_str} --> {end_str}\n{segment['text'].strip()}\n\n"
            entry_index += 1
        duration = Processor.get_video_duration(video_file)
        if duration > 0:
            total_offset_seconds += duration
            logger.info(f"视频 '{video_file.name}' 处理完成，时长: {duration:.2f}s。累计时长: {total_offset_seconds:.2f}s。")
        current_progress = 5 + int(transcribe_progress_total_weight * (i + 1) / len(video_files))
        progress_callback(current_progress)
    with open(combined_srt_path, 'w', encoding='utf-8') as f:
        f.write(combined_srt_content)
    logger.info(f"所有视频的SRT已合并到: {combined_srt_path.name}")

class Worker(QObject):
    progress = pyqtSignal(int)
    finished = pyqtSignal()
    error = pyqtSignal(str)
    log_received = pyqtSignal(str)
    def __init__(self, params: dict):
        super().__init__()
        self.params = params
        self.process = None
        self._is_running = True
    def run(self):
        queue = multiprocessing.Manager().Queue()
        self.process = multiprocessing.Process(target=run_processing_task, args=(queue, self.params))
        self.process.start()
        while self._is_running and (self.process.is_alive() or not queue.empty()):
            try:
                message_type, data = queue.get(timeout=0.1)
                if message_type == 'progress':
                    self.progress.emit(data)
                elif message_type == 'error':
                    self.error.emit(data)
                elif message_type == 'finished':
                    self.finished.emit()
                    self._is_running = False
                elif message_type == 'log':
                    self.log_received.emit(data)
            except Empty:
                continue
        if self.process:
            self.process.join()
    def stop(self):
        self._is_running = False
        if self.process and self.process.is_alive():
            self.process.terminate()
            self.process.join()

class MainWindow(QMainWindow):
    log_signal = pyqtSignal(str)
    def __init__(self):
        super().__init__()
        self.thread, self.worker = None, None
        self.config_path = "config.ini"
        self.config = configparser.ConfigParser()
        self.output_dir_name = "output_clips"
        self.supported_languages = {"中文": "Chinese", "英文": "English", "日文": "Japanese", "自动检测": None}
        self.whisper_models = {
            "快速 (tiny)": "tiny",
            "基础 (base)": "base",
            "标准 (small)": "small",
            "精准 (medium)": "medium",
            "最高精度 (large-v3)": "large-v3"
        }
        self.initUI()
        self.load_config()
        self.log_signal.connect(self.append_log)
        self.setup_logging()
    def initUI(self):
        self.setWindowTitle(f"{Config.APP_NAME} v{Config.APP_VERSION}")
        self.setGeometry(200, 200, 800, 800)
        if resources:
            self.setWindowIcon(QIcon(":/logo.png"))
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        config_group = QGroupBox("核心配置")
        config_layout = QVBoxLayout()
        input_layout = QHBoxLayout()
        self.input_path_edit = QLineEdit()
        self.input_path_edit.setReadOnly(True)
        self.input_path_edit.setPlaceholderText("请选择要处理的视频文件或目录...")
        self.browse_button = QPushButton("浏览...")
        self.browse_button.clicked.connect(self.browse_input)
        input_layout.addWidget(self.input_path_edit)
        input_layout.addWidget(self.browse_button)
        config_layout.addLayout(input_layout)
        api_layout = QFormLayout()
        self.api_key_edit = QLineEdit()
        self.api_key_edit.setEchoMode(QLineEdit.Password)
        self.language_combo = QComboBox()
        self.language_combo.addItems(self.supported_languages.keys())
        self.whisper_model_combo = QComboBox()
        self.whisper_model_combo.addItems(self.whisper_models.keys())
        api_layout.addRow("千问（通义） API Key:", self.api_key_edit)
        api_layout.addRow("语音识别语言:", self.language_combo)
        api_layout.addRow("语音识别模型:", self.whisper_model_combo)
        config_layout.addLayout(api_layout)
        config_group.setLayout(config_layout)
        main_layout.addWidget(config_group)
        param_group = QGroupBox("剪辑参数")
        param_layout = QFormLayout()
        self.num_clips_spinbox = QSpinBox()
        self.num_clips_spinbox.setRange(1, 100)
        self.num_clips_spinbox.setValue(10)
        self.clip_duration_spinbox = QSpinBox()
        self.clip_duration_spinbox.setRange(10, 600)
        self.clip_duration_spinbox.setValue(30)
        self.clip_duration_spinbox.setSuffix(" 秒")
        self.max_tasks_spinbox = QSpinBox()
        self.max_tasks_spinbox.setRange(1, 20)
        self.max_tasks_spinbox.setValue(10)
        self.power_words_edit = QLineEdit()
        self.power_words_edit.setPlaceholderText("多个词用逗号或空格隔开，留空则不使用关键词排序")
        default_keywords = "财报,利润,估值,风险,策略,逻辑,内幕,股价,市场,投资,交易,主力,资本,杠杆,预期,博弈"
        self.power_words_edit.setText(default_keywords)
        self.burn_subtitle_checkbox = QCheckBox("添加字幕")
        # self.burn_subtitle_checkbox.setChecked(True)
        param_layout.addRow("生成片段数量:", self.num_clips_spinbox)
        param_layout.addRow("目标片段时长:", self.clip_duration_spinbox)
        param_layout.addRow("高光关键词:", self.power_words_edit)
        param_layout.addRow("最大并发数:", self.max_tasks_spinbox)
        param_layout.addRow(self.burn_subtitle_checkbox)
        param_group.setLayout(param_layout)
        main_layout.addWidget(param_group)
        action_button_layout = QHBoxLayout()
        self.start_button = QPushButton("🚀 开始生成")
        self.start_button.setObjectName("PrimaryButton")
        self.start_button.setFixedHeight(45)
        self.start_button.clicked.connect(self.start_processing)
        self.stop_button = QPushButton("🛑 停止")
        self.stop_button.setFixedHeight(45)
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_processing)
        self.open_dir_button = QPushButton("📂 打开片段目录")
        self.open_dir_button.setFixedHeight(45)
        self.open_dir_button.clicked.connect(self.open_output_directory)
        action_button_layout.addWidget(self.start_button)
        action_button_layout.addWidget(self.stop_button)
        action_button_layout.addWidget(self.open_dir_button)
        main_layout.addLayout(action_button_layout)
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        main_layout.addWidget(self.progress_bar)
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setPlaceholderText("运行日志将显示在这里...")
        main_layout.addWidget(self.log_output)
    def setup_logging(self):
        if any(isinstance(h, PyQtLogHandler) for h in logger.handlers):
            return
        handler = PyQtLogHandler(self.log_signal)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    def load_config(self):
        self.config.read(self.config_path, encoding='utf-8')
        self.api_key_edit.setText(self.config.get('api', 'api_key', fallback=''))
        self.max_tasks_spinbox.setValue(self.config.getint('settings', 'max_concurrent_tasks', fallback=10))
        last_lang = self.config.get('settings', 'language', fallback='中文')
        self.language_combo.setCurrentText(last_lang)
        self.power_words_edit.setText(self.config.get('settings', 'power_words', fallback="财报,利润,估值,风险,策略,逻辑,内幕,股价,市场,投资,交易,主力,资本,杠杆,预期,博弈"))
        saved_model_name = self.config.get('settings', 'whisper_model', fallback='medium')
        display_name_to_set = "精准 (medium)"
        for display_name, model_name in self.whisper_models.items():
            if model_name == saved_model_name:
                display_name_to_set = display_name
                break
        self.whisper_model_combo.setCurrentText(display_name_to_set)
        logger.info(f"已从 {self.config_path} 加载配置。")
    def save_config(self):
        if not self.config.has_section('api'):
            self.config.add_section('api')
        if not self.config.has_section('settings'):
            self.config.add_section('settings')
        self.config.set('api', 'api_key', self.api_key_edit.text())
        self.config.set('api', 'provider', 'dashscope')
        self.config.set('api', 'model_name', 'qwen-turbo')
        self.config.set('settings', 'max_concurrent_tasks', str(self.max_tasks_spinbox.value()))
        self.config.set('settings', 'language', self.language_combo.currentText())
        self.config.set('settings', 'power_words', self.power_words_edit.text())
        selected_model_display = self.whisper_model_combo.currentText()
        model_name = self.whisper_models.get(selected_model_display, 'medium')
        self.config.set('settings', 'whisper_model', model_name)
        with open(self.config_path, 'w', encoding='utf-8') as configfile:
            self.config.write(configfile)
        logger.info(f"配置已保存到 {self.config_path}。")
    def browse_input(self):
        choice_box = QMessageBox(self)
        choice_box.setWindowTitle("选择输入类型")
        choice_box.setText("您希望处理单个视频文件，还是包含多个视频的整个目录？")
        choice_box.setStyleSheet(STYLESHEET)
        file_button = choice_box.addButton("选择单个文件 (电影)", QMessageBox.ActionRole)
        dir_button = choice_box.addButton("选择整个目录 (连续剧)", QMessageBox.ActionRole)
        cancel_button = choice_box.addButton("取消", QMessageBox.RejectRole)
        choice_box.setDefaultButton(dir_button)
        choice_box.exec_()
        path = ""
        clicked = choice_box.clickedButton()
        if clicked == file_button:
            video_extensions = " ".join([f"*{ext}" for ext in ['.mp4', '.mkv', '.avi', '.mov', '.flv', '.wmv']])
            file_path_tuple = QFileDialog.getOpenFileName(self, "选择单个视频文件", "", f"视频文件 ({video_extensions});;所有文件 (*)")
            if file_path_tuple and file_path_tuple[0]:
                path = file_path_tuple[0]
        elif clicked == dir_button:
            dir_path = QFileDialog.getExistingDirectory(self, "选择包含视频的目录")
            if dir_path:
                path = dir_path
        if path:
            self.input_path_edit.setText(path)
            logger.info(f"已选择输入路径: {path}")
        elif clicked != cancel_button:
            logger.info("用户取消了文件/目录选择。")
    def _set_controls_enabled(self, enabled: bool):
        for widget in [self.browse_button, self.api_key_edit, self.num_clips_spinbox,
                       self.clip_duration_spinbox, self.max_tasks_spinbox, self.start_button,
                       self.open_dir_button, self.language_combo, self.power_words_edit,
                       self.burn_subtitle_checkbox, self.whisper_model_combo]:
            widget.setEnabled(enabled)
        self.stop_button.setEnabled(not enabled)
        self.start_button.setText("🚀 开始生成" if enabled else "正在处理中...")
    def start_processing(self):
        if not self.input_path_edit.text():
            QMessageBox.warning(self, "警告", "请先选择一个视频文件或目录！")
            return
        self.save_config()
        self._set_controls_enabled(False)
        self.log_output.clear()
        self.progress_bar.setValue(0)
        num_clips = self.num_clips_spinbox.value()
        tolerance = min(0.5, 0.15 + num_clips / 100)
        logger.info(f"根据生成数量 {num_clips}，自适应时长容忍度设置为: {tolerance:.2f}")
        power_words_text = self.power_words_edit.text().strip()
        power_words = [word.strip() for word in re.split(r'[\s,，]+', power_words_text) if word.strip()]
        selected_lang_text = self.language_combo.currentText()
        language_code = self.supported_languages.get(selected_lang_text)
        selected_model_display = self.whisper_model_combo.currentText()
        whisper_model_name = self.whisper_models.get(selected_model_display, 'medium')
        params = {
            'input_path': self.input_path_edit.text(),
            'num_clips': num_clips,
            'clip_duration': self.clip_duration_spinbox.value(),
            'duration_tolerance': tolerance,
            'output_dir': self.output_dir_name,
            'config_path': self.config_path,
            'language': language_code,
            'power_words': power_words,
            'burn_subtitles': self.burn_subtitle_checkbox.isChecked(),
            'whisper_model': whisper_model_name
        }
        self.thread = QThread()
        self.worker = Worker(params)
        self.worker.moveToThread(self.thread)
        self.worker.log_received.connect(self.append_log)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.on_finished)
        self.worker.error.connect(self.on_error)
        self.worker.progress.connect(self.set_progress)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()
    def stop_processing(self):
        if self.worker:
            reply = QMessageBox.question(self, '确认停止', "任务正在进行中，确定要强制停止吗？", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.Yes:
                logger.warning("用户请求停止任务，正在强制终止进程...")
                self.worker.stop()
                self.thread.quit()
                self.thread.wait()
                logger.warning("任务已强制停止。")
                self.progress_bar.setValue(0)
                self._set_controls_enabled(True)
                self.append_log("------------------ 任务已被用户手动停止 ------------------")
                self.thread, self.worker = None, None
    def open_output_directory(self):
        output_path = Path(self.output_dir_name)
        if not output_path.exists():
            output_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"输出目录 '{self.output_dir_name}' 不存在，已自动创建。")
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(output_path.resolve())))
        logger.info(f"正在打开目录: {output_path.resolve()}")
    def append_log(self, text):
        self.log_output.append(text)
        self.log_output.verticalScrollBar().setValue(self.log_output.verticalScrollBar().maximum())
    def set_progress(self, value):
        self.progress_bar.setValue(value)
    def on_finished(self):
        self.progress_bar.setValue(100)
        self._set_controls_enabled(True)
        QMessageBox.information(self, "完成", f"所有任务已处理完毕！\n片段已保存到 {self.output_dir_name} 文件夹。")
        self.thread, self.worker = None, None
    def on_error(self, error_message):
        self._set_controls_enabled(True)
        self.progress_bar.setValue(0)
        QMessageBox.critical(self, "错误", f"处理过程中发生错误:\n{error_message}")
        self.thread, self.worker = None, None
    def closeEvent(self, event):
        self.save_config()
        if self.worker:
            reply = QMessageBox.question(self, '警告', "任务正在运行中，确定要退出吗？", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.Yes:
                logger.warning("程序即将退出，正在终止后台任务...")
                self.worker.stop()
                self.thread.quit()
                self.thread.wait()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()

if __name__ == '__main__':
    multiprocessing.freeze_support()
    if getattr(sys, 'frozen', False) and sys.platform == "win32":
        multiprocessing.set_start_method('spawn')
        import os
        os.environ["PATH"] += os.pathsep + str(Path(sys.executable).parent)
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLESHEET)
    main_win = MainWindow()
    main_win.show()
    sys.exit(app.exec_())