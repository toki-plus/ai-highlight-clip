import os
import re
import sys
import time
import json
import pysrt
import ffmpeg
import random
import pickle
import pynvml
import logging
import hashlib
import whisper
import tempfile
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
                             QProgressBar, QMessageBox, QCheckBox, QLabel,
                             QDialog, QDialogButtonBox)
from PyQt5.QtCore import QObject, QThread, pyqtSignal, QUrl, Qt
from PyQt5.QtGui import QDesktopServices, QIcon
import dashscope
from dashscope import Generation

try:
    import resources
except ImportError:
    resources = None

class Config:
    APP_NAME = "颜趣AI高光剪辑工具"
    APP_VERSION = "1.6.0"

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
QLabel#RecommendationLabel { color: #FFD241; font-style: italic; }
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
    generated_cover_title: str = ""
    generated_cover_subtitle: str = ""
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
            response = Generation.call(model=self.model_name, prompt=full_prompt, stream=False, use_raw_request=False)
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

    def optimize_line_breaks(self, input_srt_path: str, output_srt_path: str, max_chars_per_line: int = 10, max_lines_per_sub: int = 2) -> bool:
        self.log_callback(f"  🔄 正在使用标准算法优化字幕: {Path(input_srt_path).name}")
        if not Path(input_srt_path).exists():
            self.log_callback(f"  🔴 错误: 字幕文件 '{input_srt_path}' 不存在。")
            return False
        punctuation_to_remove_at_start = '，。！？、,.:;!?'
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
            full_text = " ".join(full_text_list)
            if not full_text:
                self.log_callback("  ⚠️ 警告: 字幕文件内容为空。")
                with open(output_srt_path, 'w', encoding='utf-8') as f:
                    pass
                return True
            clauses = re.split(r'([，。！？、,.:;!?])', full_text)
            semantic_clauses = [clauses[i] + (clauses[i+1] if i + 1 < len(clauses) and clauses[i+1] in punctuation_to_remove_at_start else '') for i in range(0, len(clauses), 2) if clauses[i]]
            final_lines = []
            for clause in semantic_clauses:
                if not clause: continue
                while len(clause) > max_chars_per_line:
                    final_lines.append(clause[:max_chars_per_line])
                    clause = clause[max_chars_per_line:]
                if clause:
                    final_lines.append(clause)
            new_subs = pysrt.SubRipFile()
            char_offset = 0
            current_sub_lines = []

            def create_sub_from_block(lines_block, current_char_offset):
                full_line = "".join(lines_block)
                full_line = full_line.strip()
                lines_to_render = full_line.split('\n')
                processed_lines = []
                for l in lines_to_render:
                    l_stripped = l.lstrip(punctuation_to_remove_at_start)
                    l_stripped = l_stripped.lstrip()
                    processed_lines.append(l_stripped)
                new_text = '\n'.join(line for line in processed_lines if line)
                text_length_for_sub = len("".join(lines_block))
                if new_text and current_char_offset + text_length_for_sub <= len(char_timestamps):
                    start_char_idx = current_char_offset
                    end_char_idx = current_char_offset + text_length_for_sub - 1
                    start_time = char_timestamps[start_char_idx]
                    end_time = char_timestamps[end_char_idx]
                    if end_time < start_time:
                        end_time = start_time + timedelta(milliseconds=200 * len(new_text))
                    new_subs.append(pysrt.SubRipItem(index=len(new_subs) + 1, start=start_time, end=end_time, text=new_text))
                return current_char_offset + text_length_for_sub

            for line in final_lines:
                if len(current_sub_lines) < max_lines_per_sub:
                    current_sub_lines.append(line)
                else:
                    char_offset = create_sub_from_block(current_sub_lines, char_offset)
                    current_sub_lines = [line]
            if current_sub_lines:
                create_sub_from_block(current_sub_lines, char_offset)
            new_subs.save(output_srt_path, encoding='utf-8')
            self.log_callback(f"  ✅ 字幕优化完成 -> {Path(output_srt_path).name}")
            return True
        except Exception as e:
            self.log_callback(f"  🔴 处理字幕时发生严重错误: {e}\n{traceback.format_exc()}")
            return False

class AdvancedSubtitleService:
    def __init__(self, log_callback=logging.info):
        self.log_callback = log_callback

    def optimize_from_word_timestamps(
        self,
        word_segments: List[Dict[str, Any]],
        output_srt_path: str,
        max_chars_per_line: int = 10,
        max_lines_per_sub: int = 2
    ) -> bool:
        self.log_callback(f"  🔄 正在使用高级词级别时间戳算法优化字幕...")
        if not word_segments:
            self.log_callback("  ⚠️ 警告: 词级别时间戳数据为空，无法生成字幕。")
            with open(output_srt_path, 'w', encoding='utf-8') as f:
                pass
            return True
        new_subs = pysrt.SubRipFile()
        current_sub_words = []
        current_lines = []
        current_line_text = ""
        sentence_enders = "，。！？,.:;!?"
        for word_info in word_segments:
            word_text = word_info.get('word', '').strip()
            if not word_text:
                continue
            potential_line = (current_line_text + word_text).strip()
            line_break = False
            if len(potential_line) > max_chars_per_line:
                line_break = True
            elif current_line_text and any(word_text.startswith(p) for p in sentence_enders):
                line_break = True
            if line_break and current_line_text:
                current_lines.append(current_line_text)
                if len(current_lines) >= max_lines_per_sub:
                    self._create_sub_from_words(new_subs, current_sub_words, current_lines)
                    current_sub_words = []
                    current_lines = []
                current_line_text = word_text
                current_sub_words.append(word_info)
            else:
                current_line_text = potential_line
                current_sub_words.append(word_info)
        if current_line_text:
            current_lines.append(current_line_text)
        if current_sub_words:
            self._create_sub_from_words(new_subs, current_sub_words, current_lines)
        try:
            new_subs.save(output_srt_path, encoding='utf-8')
            self.log_callback(f"  ✅ 字幕精确同步优化完成 -> {Path(output_srt_path).name}")
            return True
        except Exception as e:
            self.log_callback(f"  🔴 保存优化字幕时发生错误: {e}")
            return False

    def _create_sub_from_words(self, srt_file: pysrt.SubRipFile, words: List[Dict], lines: List[str]):
        if not words or not lines:
            return
        start_time = pysrt.SubRipTime.from_ordinal(int(words[0]['start'] * 1000))
        end_time = pysrt.SubRipTime.from_ordinal(int(words[-1]['end'] * 1000))
        if end_time <= start_time:
            end_time = start_time + timedelta(milliseconds=100 * len("".join(lines)))
        text = "\n".join(line.strip() for line in lines).strip()
        text = re.sub(r'^[，。！？、,.:;!?]+', '', text)
        if text:
            item = pysrt.SubRipItem(
                index=len(srt_file) + 1,
                start=start_time,
                end=end_time,
                text=text
            )
            srt_file.append(item)

class Processor:
    @staticmethod
    def _write_text_to_temp_file(text: str) -> str:
        try:
            fd, path = tempfile.mkstemp(suffix=".txt", text=False)
            with os.fdopen(fd, 'wb') as f:
                f.write(text.encode('utf-8-sig'))
            return path
        except Exception as e:
            logger.error(f"创建临时文本文件失败: {e}")
            fd, path = tempfile.mkstemp(suffix=".txt")
            os.close(fd)
            return path

    @staticmethod
    def _escape_path_for_ffmpeg(path_str: str) -> str:
        return str(Path(path_str).resolve()).replace('\\', '/')

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
    def cut_video_clip(
        source_path: Path,
        output_path: Path,
        clip_data: Clip,
        clip_start_in_file: float,
        clip_end_in_file: float,
        convert_landscape_to_portrait: bool,
        subtitle_path: Optional[Path] = None,
        font_file: Optional[str] = None,
        add_bgm: bool = False,
        vcodec: str = 'libx264',
        preset: str = 'medium',
        crf: int = 18
    ):
        main_title_file = None
        sub_title_file = None
        try:
            if clip_start_in_file < 0 or clip_end_in_file <= clip_start_in_file:
                logger.error(f"跳过剪辑，无效的时间戳: start={clip_start_in_file:.2f}, end={clip_end_in_file:.2f} for {output_path.name}")
                return
            duration = clip_end_in_file - clip_start_in_file
            if duration <= 0:
                logger.error(f"跳过剪辑，计算出的时长为零或负数: {duration:.2f}s for {output_path.name}")
                return
            probe_info = ffmpeg.probe(str(source_path))
            video_info = next((s for s in probe_info['streams'] if s['codec_type'] == 'video'), None)
            if not video_info:
                logger.error(f"无法找到视频流: {source_path.name}")
                return
            src_width, src_height = video_info['width'], video_info['height']
            is_landscape = src_width > src_height
            ffmpeg_executable = "ffmpeg.exe" if sys.platform == "win32" else "ffmpeg"
            stream = ffmpeg.input(str(source_path), ss=clip_start_in_file, t=duration)
            video_stream = stream.video
            audio_stream = stream.audio
            can_use_filters = vcodec != 'copy'
            color_palette = ["#FF6A41", "#FF7841", "#FF8C41", "#FF9A41", "#FFA841", "#FFB641", "#FFC441", "#FFD241", "#FFE041", "#FFEC41"]
            font_dir_path = Path.cwd() / "fonts"
            if not is_landscape or convert_landscape_to_portrait:
                logger.info(f"为 {output_path.name} 应用竖屏(9:16)处理流程。")
                width, height = 1080, 1920
                if can_use_filters:
                    video_stream = (
                        video_stream
                        .filter('scale', width, height, force_original_aspect_ratio='decrease')
                        .filter('pad', width, height, x='(ow-iw)/2', y='(oh-ih)/2', color='black')
                        .filter('setsar', 1)
                    )
                if can_use_filters and font_dir_path.is_dir() and font_file and clip_data.generated_cover_title:
                    main_title_size, sub_title_size = 96, 72
                    font_path_escaped = Processor._escape_path_for_ffmpeg(str(font_dir_path / font_file))
                    title_color = random.choice(color_palette)
                    main_title_file = Processor._write_text_to_temp_file(clip_data.generated_cover_title)
                    video_stream = video_stream.drawtext(
                        textfile=Processor._escape_path_for_ffmpeg(main_title_file),
                        reload=1, fontfile=font_path_escaped,
                        fontsize=main_title_size, fontcolor=title_color,
                        x='(w-text_w)/2', y='h*0.22'
                    )
                    if clip_data.generated_cover_subtitle:
                        sub_title_file = Processor._write_text_to_temp_file(clip_data.generated_cover_subtitle)
                        video_stream = video_stream.drawtext(
                            textfile=Processor._escape_path_for_ffmpeg(sub_title_file),
                            reload=1, fontfile=font_path_escaped,
                            fontsize=sub_title_size, fontcolor=title_color,
                            x='(w-text_w)/2', y=f'h*0.22 + {main_title_size}*1.2'
                        )
                if can_use_filters and subtitle_path and subtitle_path.exists() and subtitle_path.stat().st_size > 0 and font_file:
                    sub_color_hex = random.choice(color_palette)
                    ass_color_bgr = f"{sub_color_hex[5:7]}{sub_color_hex[3:5]}{sub_color_hex[1:3]}"
                    style = f"FontName={Path(font_file).stem},PrimaryColour=&H00{ass_color_bgr},BorderStyle=1,Outline=2,Alignment=2,MarginV=65"
                    video_stream = ffmpeg.filter(
                        video_stream, 'subtitles',
                        filename=Processor._escape_path_for_ffmpeg(str(subtitle_path)),
                        force_style=style,
                        fontsdir=Processor._escape_path_for_ffmpeg(str(font_dir_path))
                    )
            else:
                logger.info(f"为 {output_path.name} 应用保持横屏宽高比(标题字幕烧录在画面上)的处理流程。")
                if can_use_filters and font_dir_path.is_dir() and font_file and clip_data.generated_cover_title:
                    main_title_size = int(src_height * 0.06)
                    sub_title_size = int(src_height * 0.04)
                    font_path_escaped = Processor._escape_path_for_ffmpeg(str(font_dir_path / font_file))
                    title_color = random.choice(color_palette)
                    border_width = max(2, int(main_title_size / 20))
                    main_title_y_pos = 'h*0.1' # 距离顶部10%的高度
                    main_title_file = Processor._write_text_to_temp_file(clip_data.generated_cover_title)
                    video_stream = video_stream.drawtext(
                        textfile=Processor._escape_path_for_ffmpeg(main_title_file),
                        reload=1, fontfile=font_path_escaped,
                        fontsize=main_title_size, fontcolor=title_color,
                        x='(w-text_w)/2', y=main_title_y_pos,
                        borderw=border_width, bordercolor='black', # 添加描边
                        shadowx=2, shadowy=2, shadowcolor='black@0.6' # 添加阴影
                    )
                    if clip_data.generated_cover_subtitle:
                        sub_border_width = max(2, int(sub_title_size / 20))
                        sub_title_y_pos = f'{main_title_y_pos} + {main_title_size}*1.2'
                        sub_title_file = Processor._write_text_to_temp_file(clip_data.generated_cover_subtitle)
                        video_stream = video_stream.drawtext(
                            textfile=Processor._escape_path_for_ffmpeg(sub_title_file),
                            reload=1, fontfile=font_path_escaped,
                            fontsize=sub_title_size, fontcolor=title_color,
                            x='(w-text_w)/2', y=sub_title_y_pos,
                            borderw=sub_border_width, bordercolor='black',
                            shadowx=2, shadowy=2, shadowcolor='black@0.6'
                        )
                if can_use_filters and subtitle_path and subtitle_path.exists() and subtitle_path.stat().st_size > 0 and font_file:
                    sub_color_hex = random.choice(color_palette)
                    ass_color_bgr = f"{sub_color_hex[5:7]}{sub_color_hex[3:5]}{sub_color_hex[1:3]}"
                    subtitle_font_size = int(src_height * 0.02)
                    subtitle_margin_v = int(src_height * 0.02)
                    style = (f"FontName={Path(font_file).stem},"
                             f"FontSize={subtitle_font_size},"
                             f"PrimaryColour=&H00{ass_color_bgr},"
                             f"BorderStyle=1,Outline=2,Shadow=1," # 加粗边框和阴影
                             f"Alignment=2," # 2=底部居中
                             f"MarginV={subtitle_margin_v}") # 距离底部的边距
                    video_stream = ffmpeg.filter(
                        video_stream, 'subtitles',
                        filename=Processor._escape_path_for_ffmpeg(str(subtitle_path)),
                        force_style=style,
                        fontsdir=Processor._escape_path_for_ffmpeg(str(font_dir_path))
                    )
            if add_bgm:
                bgms_dir = Path.cwd() / "bgms"
                if bgms_dir.is_dir():
                    bgm_files = list(bgms_dir.glob("*.mp3")) + list(bgms_dir.glob("*.wav"))
                    if bgm_files:
                        bgm_path = random.choice(bgm_files)
                        bgm_stream = (
                            ffmpeg.input(str(bgm_path), stream_loop=-1)
                            .audio.filter('atrim', duration=duration)
                            .filter('volume', 0.2)
                        )
                        audio_stream = ffmpeg.filter([audio_stream, bgm_stream], 'amix', inputs=2)
                        logger.info(f"已为片段添加BGM: {bgm_path.name}")
                    else:
                        logger.warning(f"勾选了添加BGM，但 '{bgms_dir}' 文件夹为空。")
            output_options = {
                'vcodec': vcodec, 'acodec': 'aac', 'audio_bitrate': '192k', 'strict': 'experimental'
            }
            if vcodec != 'copy':
                output_options['preset'] = preset
                output_options['crf'] = crf
            else:
                 logger.warning(f"使用 'copy' 编码器，将跳过所有视频滤镜（缩放、填充、标题、字幕）。")
            (
                ffmpeg.output(video_stream, audio_stream, str(output_path), **output_options)
                .run(cmd=ffmpeg_executable, overwrite_output=True, quiet=True)
            )
            logger.info(f"✅ 成功生成片段: {output_path.name} (时长: {duration:.2f}s)")
        except ffmpeg.Error as e:
            logger.error(f"FFmpeg 剪辑失败: {output_path.name}\n{e.stderr.decode('utf-8', 'replace') if e.stderr else 'N/A'}")
        except Exception as e:
            logger.error(f"剪辑视频时发生未知错误: {e}\n{traceback.format_exc()}")
        finally:
            if main_title_file and os.path.exists(main_title_file):
                os.remove(main_title_file)
            if sub_title_file and os.path.exists(sub_title_file):
                os.remove(sub_title_file)

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
        clip_batches = self._batch_list(clips, 10)
        clips_by_id = {c.id: c for c in clips}
        total_batches = len(clip_batches)
        logger.info(f"片段将被分为 {total_batches} 个批次并发生成标题。")
        start_progress, weight, completed_batches = 65, 15, 0
        with ThreadPoolExecutor(max_workers=self.max_concurrent_tasks) as executor:
            batch_inputs = [
                {"clips": [{"id": c.id, "content": " ".join(c.content), "recommend_reason": c.recommend_reason} for c in batch]}
                for batch in clip_batches
            ]
            future_to_index = {
                executor.submit(self.llm.call_with_retry, self.prompts['标题生成'], batch_input): i
                for i, batch_input in enumerate(batch_inputs)
            }
            for future in as_completed(future_to_index):
                batch_index = future_to_index[future]
                completed_batches += 1
                try:
                    response = future.result()
                    if response and (titles_data := self.llm.parse_json_response(response)) and isinstance(titles_data, dict):
                        for clip_id, title_info in titles_data.items():
                            if clip_id in clips_by_id and isinstance(title_info, dict):
                                clips_by_id[clip_id].generated_title = title_info.get("title", "未生成标题")
                                clips_by_id[clip_id].generated_cover_title = title_info.get("cover_title", "")
                                clips_by_id[clip_id].generated_cover_subtitle = title_info.get("cover_subtitle", "")
                        logger.info(f"批次 {batch_index + 1}/{total_batches} 标题生成成功。")
                    else:
                        logger.warning(f"无法解析或API调用失败，批次 {batch_index + 1} 的标题响应。")
                except Exception as exc:
                    logger.error(f"处理标题生成批次 {batch_index + 1} 时发生异常: {exc}")
                finally:
                    self._emit_progress(start_progress, completed_batches, total_batches, weight)
        self._emit_progress(start_progress, total_batches, total_batches, weight)
        return list(clips_by_id.values())

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
        cache_dir = Path.cwd() / "cache"
        cache_dir.mkdir(exist_ok=True)
        progress_callback(5)
        video_files = _get_video_files(input_path)
        if not video_files:
            raise FileNotFoundError("在指定路径下未找到有效的视频文件。")
        use_precise_subtitles = params.get('precise_subtitles', False)
        cache_key_str = "".join([f"{p.resolve()}{p.stat().st_mtime}" for p in video_files]) + str(use_precise_subtitles)
        cache_key = hashlib.md5(cache_key_str.encode()).hexdigest()
        cache_file = cache_dir / f"{cache_key}.json"
        all_potential_clips = []
        if cache_file.exists():
            logger.info(f"🎉 发现AI分析结果缓存: {cache_file.name}，正在加载...")
            with open(cache_file, 'r', encoding='utf-8') as f:
                cached_data = json.load(f)
                all_potential_clips = [Clip.model_validate(item) for item in cached_data]
            logger.info("✅ 缓存加载成功，跳过AI分析步骤。")
            progress_callback(80)
        else:
            logger.info("未找到有效的AI分析缓存，将执行完整分析流程。")
            combined_srt_name = f"{cache_key}_combined.srt"
            srt_path = temp_dir / combined_srt_name
            _generate_and_combine_srts(video_files, srt_path, temp_dir, params, progress_callback)
            progress_callback(30)
            pipeline = AIPipeline(config, temp_dir, params, progress_signal=progress_callback)
            all_potential_clips = pipeline.run(srt_path)
            if all_potential_clips:
                with open(cache_file, 'w', encoding='utf-8') as f:
                    json.dump([clip.model_dump() for clip in all_potential_clips], f, ensure_ascii=False, indent=2)
                logger.info(f"✅ AI分析结果已缓存至: {cache_file.name}")
        if not all_potential_clips:
            raise RuntimeError("AI分析未能生成任何候选片段。")
        logger.info(f"AI分析完成，共生成 {len(all_potential_clips)} 个候选片段。")
        num_clips = params['num_clips']
        logger.info(f"开始筛选片段：数量={num_clips}")
        power_words = params.get('power_words', [])
        def is_overlapping(clip1, clip2, threshold=0.8):
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
                    logger.warning("勾选了添加标题和字幕,但 'fonts' 文件夹为空。将不添加。")
                    burn_subtitles = False
            else:
                logger.warning("勾选了添加标题和字幕,但 'fonts' 文件夹不存在。将不添加。")
                burn_subtitles = False
        progress_callback(80)
        all_word_segments = []
        if burn_subtitles and use_precise_subtitles:
            word_segments_path = temp_dir / f"{cache_key}_word_segments.pkl"
            if word_segments_path.exists():
                logger.info("正在加载精确的词级别时间戳数据用于字幕生成...")
                with open(word_segments_path, 'rb') as f:
                    all_word_segments = pickle.load(f)
            else:
                logger.warning("未找到词级别时间戳数据，将回退到标准字幕模式。")
                use_precise_subtitles = False
        srt_combined_path = temp_dir / f"{cache_key}_combined.srt"
        srt_data = []
        if burn_subtitles and not use_precise_subtitles:
            if srt_combined_path.exists():
                srt_data = Processor.parse_srt(srt_combined_path)
            else:
                logger.warning("未找到合并的SRT文件，无法生成标准字幕。")
        add_bgm_enabled = params.get('add_bgm', False)
        vcodec = params.get('vcodec', 'h264_nvenc' if params.get('use_gpu') else 'libx264')
        default_preset = 'p4' if vcodec in ['h264_nvenc', 'hevc_nvenc'] else 'medium'
        preset = params.get('preset', default_preset)
        crf = params.get('crf', 18)
        max_subtitle_lines = params.get('max_subtitle_lines', 2)
        max_chars_per_line = params.get('max_chars_per_line', 10)
        convert_landscape = params.get('convert_landscape_to_portrait', True)
        for i, clip in enumerate(final_clips):
            coarse_start_s = Processor.time_to_seconds(clip.start_time)
            coarse_end_s = Processor.time_to_seconds(clip.end_time)
            clip_subtitle_path = None
            precise_start_s, precise_end_s = coarse_start_s, coarse_end_s
            if burn_subtitles and font_file_name:
                if use_precise_subtitles and all_word_segments:
                    clip_word_segments = [word for word in all_word_segments if max(word['start'], coarse_start_s) < min(word['end'], coarse_end_s)]
                    if clip_word_segments:
                        precise_start_s = clip_word_segments[0]['start']
                        precise_end_s = clip_word_segments[-1]['end']
                        logger.info(f"片段 {i+1} 时间已精炼: [{coarse_start_s:.2f}, {coarse_end_s:.2f}] -> [{precise_start_s:.2f}, {precise_end_s:.2f}]")
                        adjusted_clip_words = []
                        for word in clip_word_segments:
                            adj_word = word.copy()
                            adj_word['start'] -= precise_start_s
                            adj_word['end'] -= precise_start_s
                            adjusted_clip_words.append(adj_word)
                        subtitle_optimizer = AdvancedSubtitleService(log_callback=logger.info)
                        optimized_clip_srt_path = temp_dir / f"optimized_clip_{i+1:02d}.srt"
                        success = subtitle_optimizer.optimize_from_word_timestamps(
                            word_segments=adjusted_clip_words, output_srt_path=str(optimized_clip_srt_path),
                            max_chars_per_line=max_chars_per_line, max_lines_per_sub=max_subtitle_lines
                        )
                        if success and optimized_clip_srt_path.exists() and optimized_clip_srt_path.stat().st_size > 0:
                            clip_subtitle_path = optimized_clip_srt_path
                        else:
                            logger.warning(f"为片段 {i+1} 生成精确同步字幕失败。")
                elif srt_data:
                    clip_srt_content, sub_index = "", 1
                    for entry in srt_data:
                        entry_start_s, entry_end_s = Processor.time_to_seconds(entry['start_time']), Processor.time_to_seconds(entry['end_time'])
                        if max(coarse_start_s, entry_start_s) < min(coarse_end_s, entry_end_s):
                            new_start_s = max(0, entry_start_s - coarse_start_s)
                            new_end_s = max(0, entry_end_s - coarse_start_s)
                            if new_end_s > new_start_s:
                                clip_srt_content += f"{sub_index}\n{Processor.seconds_to_time_str(new_start_s)} --> {Processor.seconds_to_time_str(new_end_s)}\n{entry['text']}\n\n"
                                sub_index += 1
                    if clip_srt_content:
                        raw_clip_srt_path = temp_dir / f"raw_clip_{i+1:02d}.srt"
                        with open(raw_clip_srt_path, 'w', encoding='utf-8') as f: f.write(clip_srt_content)
                        if raw_clip_srt_path.stat().st_size > 0:
                            subtitle_optimizer = SubtitleService(log_callback=logger.info)
                            optimized_clip_srt_path = temp_dir / f"optimized_clip_{i+1:02d}.srt"
                            success = subtitle_optimizer.optimize_line_breaks(
                                str(raw_clip_srt_path), str(optimized_clip_srt_path),
                                max_chars_per_line=max_chars_per_line, max_lines_per_sub=max_subtitle_lines
                            )
                            clip_subtitle_path = optimized_clip_srt_path if success else raw_clip_srt_path
            current_offset, source_file_found = 0.0, False
            for video_file in video_files:
                video_duration = Processor.get_video_duration(video_file)
                if precise_start_s < current_offset + video_duration:
                    clip_start_in_file = precise_start_s - current_offset
                    clip_end_in_file = precise_end_s - current_offset
                    safe_title = re.sub(r'[\\/*?:"<>|]', "", clip.generated_title)
                    output_filename = f"{i+1:02d}_{safe_title[:50]}.mp4"
                    Processor.cut_video_clip(
                        source_path=video_file, output_path=output_dir / output_filename,
                        clip_data=clip, clip_start_in_file=clip_start_in_file,
                        clip_end_in_file=clip_end_in_file,
                        convert_landscape_to_portrait=convert_landscape,
                        subtitle_path=clip_subtitle_path,
                        font_file=font_file_name, add_bgm=add_bgm_enabled,
                        vcodec=vcodec, preset=preset, crf=crf
                    )
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

def _get_video_files(input_path: Path) -> List[Path]:
    VIDEO_EXTENSIONS = ['.mp4', '.mkv', '.avi', '.mov', '.flv', '.wmv']
    if input_path.is_dir():
        files = sorted([p for p in input_path.iterdir() if p.suffix.lower() in VIDEO_EXTENSIONS])
        return files
    elif input_path.is_file() and input_path.suffix.lower() in VIDEO_EXTENSIONS:
        return [input_path]
    return []

def _generate_and_combine_srts(video_files: List[Path], combined_srt_path: Path, temp_dir: Path, params: dict, progress_callback):
    language_to_use = params.get('language')
    whisper_model_name = params.get('whisper_model', 'medium')
    use_precise_subtitles = params.get('precise_subtitles', False)
    logger.info(f"正在加载 Whisper 模型: {whisper_model_name}...")
    whisper_model = whisper.load_model(whisper_model_name)
    logger.info("Whisper 模型加载完毕。")
    total_offset_seconds, combined_srt_content, entry_index = 0.0, "", 1
    all_word_segments = []
    transcribe_progress_total_weight = 25
    for i, video_file in enumerate(video_files):
        logger.info(f"--- 开始处理视频 {i+1}/{len(video_files)}: {video_file.name} ---")
        result = None
        individual_word_segments_path = temp_dir / f"{video_file.stem}_word_segments.pkl"
        individual_srt_path = temp_dir / f"{video_file.stem}.srt"
        if use_precise_subtitles and individual_word_segments_path.exists():
            logger.info(f"发现已存在的词级别时间戳缓存: '{individual_word_segments_path.name}'，将直接加载。")
            with open(individual_word_segments_path, 'rb') as f: result = pickle.load(f)
        elif not use_precise_subtitles and individual_srt_path.exists():
            logger.info(f"发现已存在的SRT文件: '{individual_srt_path.name}'，将直接使用。")
            result = {'segments': [{'start': Processor.time_to_seconds(e['start_time']), 'end': Processor.time_to_seconds(e['end_time']), 'text': e['text']} for e in Processor.parse_srt(individual_srt_path)]}
        if not result:
            log_msg = f"正在进行语音识别 (语言: {language_to_use or '自动检测'})"
            if use_precise_subtitles: log_msg += ", 开启词级别时间戳"
            logger.info(log_msg + "...")
            transcribe_args = {'language': language_to_use} if language_to_use else {}
            if use_precise_subtitles: transcribe_args['word_timestamps'] = True
            result = whisper_model.transcribe(str(video_file), verbose=False, **transcribe_args)
            if use_precise_subtitles and 'segments' in result:
                with open(individual_word_segments_path, 'wb') as f: pickle.dump(result, f)
                logger.info(f"已为 '{video_file.name}' 生成并缓存词级别时间戳数据。")
            individual_srt_content = "".join(f"{seg_idx+1}\n{Processor.seconds_to_time_str(segment['start'])} --> {Processor.seconds_to_time_str(segment['end'])}\n{segment['text'].strip()}\n\n" for seg_idx, segment in enumerate(result['segments']))
            with open(individual_srt_path, 'w', encoding='utf-8') as f: f.write(individual_srt_content)
            logger.info(f"已为 '{video_file.name}' 生成独立的SRT文件: {individual_srt_path.name}")
        if use_precise_subtitles and 'segments' in result:
            for segment in result['segments']:
                if 'words' in segment:
                    for word_info in segment['words']:
                        if 'start' in word_info and 'end' in word_info:
                            adjusted_word_info = word_info.copy()
                            adjusted_word_info['start'] += total_offset_seconds
                            adjusted_word_info['end'] += total_offset_seconds
                            all_word_segments.append(adjusted_word_info)
        for segment in result['segments']:
            start_str = Processor.seconds_to_time_str(segment['start'] + total_offset_seconds)
            end_str = Processor.seconds_to_time_str(segment['end'] + total_offset_seconds)
            combined_srt_content += f"{entry_index}\n{start_str} --> {end_str}\n{segment['text'].strip()}\n\n"
            entry_index += 1
        duration = Processor.get_video_duration(video_file)
        if duration > 0: total_offset_seconds += duration
        logger.info(f"视频 '{video_file.name}' 处理完成，时长: {duration:.2f}s。累计时长: {total_offset_seconds:.2f}s。")
        current_progress = 5 + int(transcribe_progress_total_weight * (i + 1) / len(video_files))
        progress_callback(current_progress)
    with open(combined_srt_path, 'w', encoding='utf-8') as f: f.write(combined_srt_content)
    logger.info(f"所有视频的SRT已合并到: {combined_srt_path.name}")
    if use_precise_subtitles:
        combined_word_segments_path = temp_dir / f"{combined_srt_path.stem}_word_segments.pkl"
        with open(combined_word_segments_path, 'wb') as f: pickle.dump(all_word_segments, f)
        logger.info(f"所有视频的词级别时间戳数据已合并到: {combined_word_segments_path.name}")

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

class DurationCalculator(QObject):
    duration_calculated = pyqtSignal(float)
    def __init__(self, path_str):
        super().__init__()
        self.path = Path(path_str)
    def run(self):
        total_duration = 0.0
        try:
            video_files = _get_video_files(self.path)
            for video_file in video_files:
                total_duration += Processor.get_video_duration(video_file)
        except Exception as e:
            logger.error(f"计算总时长时出错: {e}")
            total_duration = 0.0
        self.duration_calculated.emit(total_duration)

class OrientationChecker(QObject):
    orientation_checked = pyqtSignal(bool, bool)

    def __init__(self, path_str):
        super().__init__()
        self.path = Path(path_str)

    def run(self):
        has_landscape = False
        has_portrait = False
        try:
            video_files = _get_video_files(self.path)
            if not video_files:
                self.orientation_checked.emit(False, False)
                return
            for video_file in video_files:
                try:
                    probe = ffmpeg.probe(str(video_file))
                    video_stream = next((s for s in probe['streams'] if s['codec_type'] == 'video'), None)
                    if video_stream:
                        width = video_stream.get('width', 0)
                        height = video_stream.get('height', 0)
                        if width > height:
                            has_landscape = True
                            logger.info(f"检测到横屏视频: {video_file.name}")
                        else:
                            has_portrait = True
                            logger.info(f"检测到竖屏或方形视频: {video_file.name}")
                except Exception as e:
                    logger.warning(f"检查视频 '{video_file.name}' 方向时出错: {e}")
                    continue
                if has_landscape and has_portrait:
                    break
        except Exception as e:
            logger.error(f"检测视频方向时发生未知错误: {e}")
        self.orientation_checked.emit(has_landscape, has_portrait)

class MainWindow(QMainWindow):
    log_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.thread, self.worker = None, None
        self.duration_thread = None
        self.orientation_check_thread = None
        self.total_video_duration_seconds = 0
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
        self.vcodecs_cpu = ['libx264', 'libx265', 'copy']
        self.vcodecs_gpu = ['h264_nvenc', 'hevc_nvenc']
        self.presets = {
            'libx264': ['veryslow', 'slow', 'medium', 'fast', 'veryfast', 'ultrafast'],
            'libx265': ['veryslow', 'slow', 'medium', 'fast', 'veryfast', 'ultrafast'],
            'h264_nvenc': ['p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7'],
            'hevc_nvenc': ['p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7'],
            'copy': []
        }
        self.has_gpu = False
        try:
            pynvml.nvmlInit()
            if pynvml.nvmlDeviceGetCount() > 0:
                self.has_gpu = True
                logger.info("检测到 NVIDIA GPU，将启用GPU加速选项。")
            else:
                logger.warning("未检测到 NVIDIA GPU，将使用CPU进行编码。")
        except pynvml.NVMLError:
            logger.warning("pynvml 初始化失败。无法检测NVIDIA GPU，将使用CPU进行编码。")
        self.initUI()
        self.load_config()
        self.log_signal.connect(self.append_log)
        self.setup_logging()

    def initUI(self):
        self.setWindowTitle(f"{Config.APP_NAME} v{Config.APP_VERSION}")
        self.setGeometry(100, 100, 1200, 800)
        if resources:
            self.setWindowIcon(QIcon(":/logo.png"))
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_horizontal_layout = QHBoxLayout(central_widget)
        left_column_layout = QVBoxLayout()
        config_group = QGroupBox("核心配置")
        api_layout = QFormLayout()
        self.input_path_edit = QLineEdit()
        self.input_path_edit.setReadOnly(True)
        self.input_path_edit.setPlaceholderText("请选择要处理的视频文件或目录...")
        self.browse_button = QPushButton("浏览...")
        self.browse_button.clicked.connect(self.browse_input)
        input_layout = QHBoxLayout()
        input_layout.addWidget(self.input_path_edit)
        input_layout.addWidget(self.browse_button)
        api_layout.addRow(input_layout)
        self.api_key_edit = QLineEdit()
        self.api_key_edit.setEchoMode(QLineEdit.Password)
        self.language_combo = QComboBox()
        self.language_combo.addItems(self.supported_languages.keys())
        self.whisper_model_combo = QComboBox()
        self.whisper_model_combo.addItems(self.whisper_models.keys())
        api_layout.addRow("千问（通义） API Key:", self.api_key_edit)
        api_layout.addRow("语音识别语言:", self.language_combo)
        api_layout.addRow("语音识别模型:", self.whisper_model_combo)
        config_group.setLayout(api_layout)
        left_column_layout.addWidget(config_group)
        param_group = QGroupBox("剪辑参数")
        param_layout = QFormLayout()
        self.num_clips_spinbox = QSpinBox()
        self.num_clips_spinbox.setRange(1, 999)
        self.num_clips_spinbox.setValue(10)
        self.num_clips_spinbox.valueChanged.connect(self.update_recommendation_text)
        self.clip_duration_spinbox = QSpinBox()
        self.clip_duration_spinbox.setRange(10, 600)
        self.clip_duration_spinbox.setValue(30)
        self.clip_duration_spinbox.setSuffix(" 秒")
        self.clip_duration_spinbox.valueChanged.connect(self.update_recommendation_text)
        self.max_tasks_spinbox = QSpinBox()
        self.max_tasks_spinbox.setRange(1, 20)
        self.max_tasks_spinbox.setValue(10)
        self.power_words_edit = QLineEdit()
        self.power_words_edit.setPlaceholderText("多个词用逗号或空格隔开，留空则不使用关键词排序")
        default_keywords = "财报,利润,估值,风险,策略,逻辑,内幕,股价,市场,投资,交易,主力,资本,杠杆,预期,博弈"
        self.power_words_edit.setText(default_keywords)
        self.subtitle_lines_spinbox = QSpinBox()
        self.subtitle_lines_spinbox.setRange(1, 3)
        self.subtitle_lines_spinbox.setValue(2)
        self.max_chars_per_line_spinbox = QSpinBox()
        self.max_chars_per_line_spinbox.setRange(5, 30)
        self.max_chars_per_line_spinbox.setValue(10)
        param_layout.addRow("生成片段数量:", self.num_clips_spinbox)
        param_layout.addRow("目标片段时长:", self.clip_duration_spinbox)
        self.recommendation_label = QLabel("请先选择视频以获取建议。")
        self.recommendation_label.setObjectName("RecommendationLabel")
        self.recommendation_label.setWordWrap(True)
        param_layout.addRow("智能建议:", self.recommendation_label)
        param_layout.addRow("高光关键词:", self.power_words_edit)
        param_layout.addRow("字幕最大行数:", self.subtitle_lines_spinbox)
        param_layout.addRow("字幕每行最大字数:", self.max_chars_per_line_spinbox)
        param_layout.addRow("最大并发数:", self.max_tasks_spinbox)
        self.landscape_to_portrait_checkbox = QCheckBox("横屏转竖屏")
        self.landscape_to_portrait_checkbox.setToolTip("如果检测到横屏视频，勾选此项会将其强制转为9:16竖屏（上下加黑边）。\n不勾选则保持原始宽高比，并将标题和字幕直接烧录在视频画面上。")
        self.landscape_to_portrait_checkbox.setChecked(False)
        self.landscape_to_portrait_checkbox.setEnabled(False)
        self.burn_subtitle_checkbox = QCheckBox("添加标题和字幕")
        self.precise_subtitle_checkbox = QCheckBox("启用精准字幕")
        self.precise_subtitle_checkbox.setToolTip("推荐开启。使用词级别时间戳，口播同步更精准，但首次识别稍慢。")
        self.add_bgm_checkbox = QCheckBox("添加随机BGM")
        self.gpu_accel_checkbox = QCheckBox("启用GPU加速 (NVENC)")
        self.gpu_accel_checkbox.setEnabled(self.has_gpu)
        self.gpu_accel_checkbox.setChecked(self.has_gpu)
        self.gpu_accel_checkbox.stateChanged.connect(self.update_vcodec_options)
        checkbox_layout = QHBoxLayout()
        checkbox_layout.addWidget(self.landscape_to_portrait_checkbox)
        checkbox_layout.addWidget(self.burn_subtitle_checkbox)
        checkbox_layout.addWidget(self.precise_subtitle_checkbox)
        checkbox_layout.addWidget(self.add_bgm_checkbox)
        checkbox_layout.addWidget(self.gpu_accel_checkbox)
        checkbox_layout.addStretch()
        param_layout.addRow(checkbox_layout)
        param_group.setLayout(param_layout)
        left_column_layout.addWidget(param_group)
        left_column_layout.addStretch(1)
        bottom_controls_layout = QVBoxLayout()
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
        bottom_controls_layout.addLayout(action_button_layout)
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        bottom_controls_layout.addWidget(self.progress_bar)
        left_column_layout.addLayout(bottom_controls_layout)
        right_column_layout = QVBoxLayout()
        encoding_group = QGroupBox("高级编码设置")
        encoding_form_layout = QFormLayout()
        self.help_button = QPushButton("?")
        self.help_button.setFixedSize(24, 24)
        self.help_button.setStyleSheet("font-weight: bold; border-radius: 12px;")
        self.help_button.setToolTip("点击查看关于编码设置的详细说明和推荐")
        self.help_button.clicked.connect(self.show_encoding_help)
        self.vcodec_combo = QComboBox()
        self.preset_combo = QComboBox()
        self.crf_spinbox = QSpinBox()
        self.crf_spinbox.setRange(0, 51)
        self.crf_spinbox.setValue(18)
        self.vcodec_combo.currentTextChanged.connect(self.update_preset_options)
        vcodec_label = QLabel("视频编码器:")
        preset_label = QLabel("编码预设:")
        crf_label = QLabel("质量因子 (CRF/QP):")
        help_label = QLabel("参数说明：")
        encoding_form_layout.addRow(help_label, self.help_button)
        encoding_form_layout.addRow(vcodec_label, self.vcodec_combo)
        encoding_form_layout.addRow(preset_label, self.preset_combo)
        encoding_form_layout.addRow(crf_label, self.crf_spinbox)
        encoding_group.setLayout(encoding_form_layout)
        right_column_layout.addWidget(encoding_group)
        self.update_vcodec_options()
        log_group = QGroupBox("运行日志")
        log_layout = QVBoxLayout()
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setPlaceholderText("运行日志将显示在这里...")
        log_layout.addWidget(self.log_output)
        log_group.setLayout(log_layout)
        right_column_layout.addWidget(log_group)
        left_widget = QWidget()
        left_widget.setLayout(left_column_layout)
        right_widget = QWidget()
        right_widget.setLayout(right_column_layout)
        main_horizontal_layout.addWidget(left_widget, 1)
        main_horizontal_layout.addWidget(right_widget, 1)

    def update_vcodec_options(self):
        current_vcodec = self.vcodec_combo.currentText()
        self.vcodec_combo.clear()
        if self.gpu_accel_checkbox.isChecked() and self.has_gpu:
            self.vcodec_combo.addItems(self.vcodecs_gpu)
            if current_vcodec in self.vcodecs_gpu:
                self.vcodec_combo.setCurrentText(current_vcodec)
            else:
                self.vcodec_combo.setCurrentText(self.vcodecs_gpu[0])
        else:
            self.vcodec_combo.addItems(self.vcodecs_cpu)
            if current_vcodec in self.vcodecs_cpu:
                self.vcodec_combo.setCurrentText(current_vcodec)
            else:
                self.vcodec_combo.setCurrentText(self.vcodecs_cpu[0])
        new_vcodec = self.vcodec_combo.currentText()
        self.preset_combo.clear()
        presets = self.presets.get(new_vcodec, [])
        self.preset_combo.addItems(presets)
        if new_vcodec in self.vcodecs_gpu:
            if 'p4' in presets:
                self.preset_combo.setCurrentText('p4')
        elif new_vcodec in self.vcodecs_cpu:
            if 'medium' in presets:
                self.preset_combo.setCurrentText('medium')
        self.preset_combo.setEnabled(bool(presets))
        self.crf_spinbox.setEnabled(new_vcodec != 'copy')

    def update_preset_options(self, vcodec):
        current_preset = self.preset_combo.currentText()
        self.preset_combo.clear()
        presets = self.presets.get(vcodec, [])
        self.preset_combo.addItems(presets)
        if current_preset in presets:
            self.preset_combo.setCurrentText(current_preset)
        self.preset_combo.setEnabled(bool(presets))
        self.crf_spinbox.setEnabled(vcodec != 'copy')

    def show_encoding_help(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("编码设置帮助")
        dialog.setMinimumSize(800, 500)
        layout = QVBoxLayout(dialog)
        text_edit = QTextEdit()
        text_edit.setReadOnly(True)
        help_text = """
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body { font-family: "Segoe UI", "Microsoft YaHei", sans-serif; background-color: #1A1D2A; color: #D0D0D0; }
                h2 { color: #25F4EE; border-bottom: 1px solid #4A4E60; padding-bottom: 5px; }
                p, li { line-height: 1.6; }
                b { color: #FE2C55; }
                code { background-color: #3B3F51; padding: 2px 6px; border-radius: 4px; font-family: "Consolas", "Courier New", monospace; }
                .recommendation { border-left: 3px solid #25F4EE; padding-left: 15px; margin-top: 10px; background-color: #24283B; }
                hr { border: none; border-top: 1px solid #3B3F51; }
            </style>
        </head>
        <body>
            <h2>高级编码设置说明</h2>
            <p>这里是视频输出的“专家模式”，允许您精细控制最终视频的<b>画质</b>、<b>文件大小</b>和<b>生成速度</b>。如果您不确定如何选择，<b>使用默认值是兼顾质量和速度的最佳选择</b>。</p>
            <hr>

            <h2>1. 视频编码器 (Video Encoder)</h2>
            <p>编码器是决定视频如何被压缩的“引擎”。不同的引擎有不同的特点：</p>
            <ul>
                <li><b><code>libx264</code> (CPU编码, H.264)</b>
                    <ul>
                        <li><b>特点:</b> 质量极高，兼容性最好。几乎所有设备和平台都能流畅播放。</li>
                        <li><b>缺点:</b> 完全依赖CPU进行计算，速度相对较慢。</li>
                        <li><b>适用场景:</b> 追求最高画质和最佳兼容性，且不介意等待更长时间的用户。</li>
                    </ul>
                </li>
                <li><b><code>libx265</code> (CPU编码, H.265/HEVC)</b>
                    <ul>
                        <li><b>特点:</b> 新一代编码标准。在肉眼几乎看不出差别的情况下，文件大小比 <code>libx264</code> 小约 <b>30-50%</b>。</li>
                        <li><b>缺点:</b> 编码过程比 <code>libx264</code> 更慢。部分较老的设备或播放器可能不支持H.265格式。</li>
                        <li><b>适用场景:</b> 对存储空间敏感，希望获得更小文件体积，且确认播放设备支持H.265的用户。</li>
                    </ul>
                </li>
                <li><b><code>h264_nvenc</code> (GPU加速, H.264)</b>
                    <ul>
                        <li><b>特点:</b> 利用NVIDIA显卡进行硬件编码，速度比CPU编码快 <b>5到10倍</b>甚至更多，能极大缩短等待时间。</li>
                        <li><b>缺点:</b> 在同等文件大小下，画质通常略逊于 <code>libx264</code> 的慢速预设，但对于短视频和社交媒体分享，这种差异几乎可以忽略。</li>
                        <li><b>适用场景:</b> <b>强烈推荐拥有NVIDIA显卡的用户使用！</b>是速度和质量的完美结合。</li>
                    </ul>
                </li>
                <li><b><code>hevc_nvenc</code> (GPU加速, H.265/HEVC)</b>
                    <ul>
                        <li><b>特点:</b> 结合了GPU的超快速度和H.265的高压缩率，能生成体积很小的视频文件。</li>
                        <li><b>缺点:</b> 同 <code>libx265</code>，需要考虑播放设备的兼容性。</li>
                        <li><b>适用场景:</b> 希望快速生成超小体积视频，并确认播放设备支持的用户。</li>
                    </ul>
                </li>
                 <li><b><code>copy</code> (无损复制)</b>
                    <ul>
                        <li><b>特点:</b> <b>完全无损</b>。它不进行任何重新编码，只是将原始视频数据原封不动地“剪切”出来。速度最快，画质与源文件100%相同。</li>
                        <li><b>重大限制:</b> 在此模式下，所有需要修改画面的操作（包括<b>添加标题、字幕、转为竖屏的黑边</b>）都将<b>完全失效</b>。</li>
                        <li><b>适用场景:</b> 仅当您需要对原始视频进行精确裁剪，且不需要任何画面修改时使用。<b>对于本工具的核心功能（生成带标题字幕的竖屏视频），此选项无效。</b></li>
                    </ul>
                </li>
            </ul>
            <div class="recommendation">
                <p><b>如何选择？</b><br>
                - 有NVIDIA显卡？ <b>首选 <code>h264_nvenc</code></b>。<br>
                - 没有显卡但追求画质？ <b>选择 <code>libx264</code></b>。<br>
                - 存储空间极其宝贵？ <b>可以尝试 <code>libx265</code> 或 <code>hevc_nvenc</code></b>。
                </p>
            </div>
            <hr>

            <h2>2. 编码预设 (Encoding Preset)</h2>
            <p>预设是您告诉编码器“可以花多少时间来思考如何压缩得更好”。越慢的预设，编码器分析得越精细，压缩效果越好（同等画质下文件更小）。</p>
            <ul>
                <li><b>对于CPU编码 (<code>libx264</code> / <code>libx265</code>):</b> 范围从 <code>veryslow</code> 到 <code>ultrafast</code>。
                    <ul>
                        <li><code>veryslow</code>: 速度最慢，压缩率最高。</li>
                        <li><code>medium</code>: 速度和压缩率的官方推荐平衡点。</li>
                        <li><code>ultrafast</code>: 速度最快，但文件会大很多。</li>
                    </ul>
                </li>
                <li><b>对于GPU编码 (<code>..._nvenc</code>):</b> 范围从 <code>p1</code> (慢) 到 <code>p7</code> (快)。
                    <ul>
                        <li><code>p1</code>-<code>p3</code> (慢速): 质量更高。</li>
                        <li><code>p4</code> (默认): 良好的平衡。</li>
                        <li><code>p5</code>-<code>p7</code> (快速): 速度更快，质量略有下降。</li>
                    </ul>
                </li>
            </ul>
             <div class="recommendation">
                <p><b>如何选择？</b><br>
                - 通常保持默认的 <code>medium</code> 或 <code>p4</code> 即可。<br>
                - 如果发现文件有点大，可以尝试更慢一档的预设（如 <code>slow</code> 或 <code>p3</code>），这会在不牺牲画质的情况下减小文件体积。
                </p>
            </div>
            <hr>

            <h2>3. 质量因子 (CRF / QP)</h2>
            <p>这是控制视频最终视觉质量的最直接参数。<b>请记住一个核心原则：数值越低，画质越高，文件越大。</b></p>
            <ul>
                <li><b>CRF (恒定速率因子) - 用于CPU编码 (<code>libx264</code> / <code>libx265</code>)</b>
                    <p>CRF模式会尽力在整个视频中保持一个恒定的“感知质量”。</p>
                    <ul>
                        <li><code>0</code>: 数学上的无损压缩。文件体积会<b>极其巨大</b>，通常仅用于专业存档，不推荐日常使用。</li>
                        <li><code>16-18</code>: <b>视觉无损范围。</b>人眼几乎无法分辨与源视频的差异，是追求高质量输出的理想选择。</li>
                        <li><code>23</code>: <b>高质量默认值。</b> 质量非常好，且文件大小适中。</li>
                        <li><code>28</code> 及以上: 画质开始出现肉眼可见的损失。</li>
                    </ul>
                </li>
                <li><b>QP (量化参数) - 用于GPU编码 (<code>..._nvenc</code>)</b>
                    <p>QP模式试图为视频的每个部分应用一个固定的量化级别，概念上与CRF类似。</p>
                     <ul>
                        <li><code>0</code>: 同样是无损，同样不推荐。</li>
                        <li><code>18-22</code>: <b>非常高的质量范围。</b></li>
                        <li><code>23-25</code>: <b>高质量范围</b>，通常是很好的平衡点。</li>
                        <li><code>28</code> 及以上: 画质下降会比较明显。</li>
                    </ul>
                </li>
            </ul>
            <div class="recommendation">
                <p><b>如何选择？</b><br>
                - 想要最好的画质？设置 <b>18</b> 或更低。<br>
                - 想要在质量和文件大小间取得完美平衡？使用默认值或在 <b>20-23</b> 之间选择。<br>
                - 想快速生成一个预览版的小样？可以设置为 <b>26-28</b>。
                </p>
            </div>
        </body>
        </html>
        """
        text_edit.setHtml(help_text)
        button_box = QDialogButtonBox(QDialogButtonBox.Ok)
        button_box.accepted.connect(dialog.accept)
        layout.addWidget(text_edit)
        layout.addWidget(button_box)
        dialog.exec_()

    def setup_logging(self):
        if any(isinstance(h, PyQtLogHandler) for h in logger.handlers):
            return
        handler = PyQtLogHandler(self.log_signal)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    def load_config(self):
        self.config.read(self.config_path, encoding='utf-8')
        self.api_key_edit.setText(self.config.get('api', 'api_key', fallback=''))
        self.num_clips_spinbox.setValue(self.config.getint('settings', 'num_clips', fallback=10))
        self.clip_duration_spinbox.setValue(self.config.getint('settings', 'clip_duration', fallback=30))
        self.max_tasks_spinbox.setValue(self.config.getint('settings', 'max_concurrent_tasks', fallback=10))
        self.subtitle_lines_spinbox.setValue(self.config.getint('settings', 'subtitle_lines', fallback=2))
        self.max_chars_per_line_spinbox.setValue(self.config.getint('settings', 'max_chars_per_line', fallback=10))
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
        self.burn_subtitle_checkbox.setChecked(self.config.getboolean('settings', 'burn_subtitles', fallback=False))
        self.precise_subtitle_checkbox.setChecked(self.config.getboolean('settings', 'precise_subtitles', fallback=True))
        self.add_bgm_checkbox.setChecked(self.config.getboolean('settings', 'add_bgm', fallback=False))
        use_gpu = self.config.getboolean('settings', 'use_gpu', fallback=self.has_gpu)
        self.gpu_accel_checkbox.setChecked(use_gpu and self.has_gpu)
        self.update_vcodec_options()
        default_vcodec = self.vcodec_combo.itemText(0) if self.vcodec_combo.count() > 0 else 'libx264'
        self.vcodec_combo.setCurrentText(self.config.get('encoding', 'vcodec', fallback=default_vcodec))
        self.update_preset_options(self.vcodec_combo.currentText())
        default_preset = 'p4' if self.vcodec_combo.currentText() in self.vcodecs_gpu else 'medium'
        self.preset_combo.setCurrentText(self.config.get('encoding', 'preset', fallback=default_preset))
        self.crf_spinbox.setValue(self.config.getint('encoding', 'crf', fallback=18))
        logger.info(f"已从 {self.config_path} 加载配置。")

    def save_config(self):
        if not self.config.has_section('api'): self.config.add_section('api')
        if not self.config.has_section('settings'): self.config.add_section('settings')
        if not self.config.has_section('encoding'): self.config.add_section('encoding')
        self.config.set('api', 'api_key', self.api_key_edit.text())
        self.config.set('api', 'provider', 'dashscope')
        self.config.set('api', 'model_name', 'qwen-plus-2025-07-28')
        self.config.set('settings', 'num_clips', str(self.num_clips_spinbox.value()))
        self.config.set('settings', 'clip_duration', str(self.clip_duration_spinbox.value()))
        self.config.set('settings', 'max_concurrent_tasks', str(self.max_tasks_spinbox.value()))
        self.config.set('settings', 'subtitle_lines', str(self.subtitle_lines_spinbox.value()))
        self.config.set('settings', 'max_chars_per_line', str(self.max_chars_per_line_spinbox.value()))
        self.config.set('settings', 'language', self.language_combo.currentText())
        self.config.set('settings', 'power_words', self.power_words_edit.text())
        selected_model_display = self.whisper_model_combo.currentText()
        model_name = self.whisper_models.get(selected_model_display, 'medium')
        self.config.set('settings', 'whisper_model', model_name)
        self.config.set('settings', 'burn_subtitles', str(self.burn_subtitle_checkbox.isChecked()))
        self.config.set('settings', 'precise_subtitles', str(self.precise_subtitle_checkbox.isChecked()))
        self.config.set('settings', 'add_bgm', str(self.add_bgm_checkbox.isChecked()))
        self.config.set('settings', 'use_gpu', str(self.gpu_accel_checkbox.isChecked()))
        self.config.set('encoding', 'vcodec', self.vcodec_combo.currentText())
        self.config.set('encoding', 'preset', self.preset_combo.currentText())
        self.config.set('encoding', 'crf', str(self.crf_spinbox.value()))
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
        self.landscape_to_portrait_checkbox.setEnabled(False)
        self.landscape_to_portrait_checkbox.setChecked(True)
        if path:
            self.input_path_edit.setText(path)
            logger.info(f"已选择输入路径: {path}")
            self.calculate_total_duration(path)
            self.check_video_orientation(path)
        elif clicked != cancel_button:
            logger.info("用户取消了文件/目录选择。")

    def check_video_orientation(self, path_str):
        logger.info("正在检测视频方向...")
        if hasattr(self, 'orientation_check_thread') and self.orientation_check_thread and self.orientation_check_thread.isRunning():
            self.orientation_check_thread.quit()
            self.orientation_check_thread.wait()
        self.orientation_check_thread = QThread()
        self.orientation_checker = OrientationChecker(path_str)
        self.orientation_checker.moveToThread(self.orientation_check_thread)
        self.orientation_checker.orientation_checked.connect(self.on_orientation_checked)
        self.orientation_check_thread.started.connect(self.orientation_checker.run)
        self.orientation_check_thread.finished.connect(self.orientation_checker.deleteLater)
        self.orientation_check_thread.finished.connect(self.orientation_check_thread.deleteLater)
        self.orientation_check_thread.start()

    def on_orientation_checked(self, has_landscape: bool, has_portrait: bool):
        if has_landscape and not has_portrait:
            logger.info("检测到输入素材全为横屏视频。用户现在可以选择是否进行转换。")
            self.landscape_to_portrait_checkbox.setEnabled(True)
            self.landscape_to_portrait_checkbox.setChecked(False)
        elif has_portrait and not has_landscape:
            logger.info("检测到输入素材全为竖屏视频。将禁用横屏转换选项。")
            self.landscape_to_portrait_checkbox.setEnabled(False)
            self.landscape_to_portrait_checkbox.setChecked(False)
        elif has_landscape and has_portrait:
            logger.warning("检测到输入素材包含横屏和竖屏混合视频。为保证处理一致性，将禁用横屏转换选项并按默认逻辑处理。")
            self.landscape_to_portrait_checkbox.setEnabled(False)
            self.landscape_to_portrait_checkbox.setChecked(False)
        else:
            logger.info("未检测到有效视频或无法确定视频方向。将禁用横屏转换选项。")
            self.landscape_to_portrait_checkbox.setEnabled(False)
            self.landscape_to_portrait_checkbox.setChecked(False)
        if hasattr(self, 'orientation_check_thread') and self.orientation_check_thread:
            self.orientation_check_thread.quit()
            self.orientation_check_thread.wait()
            self.orientation_check_thread = None
            self.orientation_checker = None

    def calculate_total_duration(self, path_str):
        self.recommendation_label.setText("正在计算视频总时长...")
        if hasattr(self, 'duration_thread') and self.duration_thread and self.duration_thread.isRunning():
            self.duration_thread.quit()
            self.duration_thread.wait()
        self.duration_thread = QThread()
        self.duration_worker = DurationCalculator(path_str)
        self.duration_worker.moveToThread(self.duration_thread)
        self.duration_worker.duration_calculated.connect(self.on_duration_calculated)
        self.duration_thread.started.connect(self.duration_worker.run)
        self.duration_thread.finished.connect(self.duration_worker.deleteLater)
        self.duration_thread.finished.connect(self.duration_thread.deleteLater)
        self.duration_thread.start()

    def on_duration_calculated(self, total_seconds):
        self.total_video_duration_seconds = total_seconds
        self.update_recommendation_text()

    def update_recommendation_text(self):
        if self.total_video_duration_seconds <= 0:
            self.recommendation_label.setText("请先选择视频以获取建议。")
            return
        num_clips = self.num_clips_spinbox.value()
        clip_duration = self.clip_duration_spinbox.value()
        total_minutes = self.total_video_duration_seconds / 60
        if clip_duration == 0: return
        low_density_ratio = 0.20
        high_density_ratio = 0.40
        recommended_low = int((self.total_video_duration_seconds * low_density_ratio) / clip_duration)
        recommended_high = int((self.total_video_duration_seconds * high_density_ratio) / clip_duration)
        recommended_low = max(1, recommended_low)
        recommended_high = max(recommended_low, recommended_high)
        if num_clips < recommended_low:
            self.recommendation_label.setText(f"设置合理。您也可以尝试生成更多片段以捕捉所有高光。")
        elif num_clips > recommended_high:
            self.recommendation_label.setText(f"注意: 数量可能偏高，会包含非高光内容。\n基于{total_minutes:.1f}分钟视频，建议生成 {recommended_low}-{recommended_high} 个高质量片段。")
        else:
            self.recommendation_label.setText(f"设置合理。将在约{total_minutes:.1f}分钟的视频中寻找高光。")

    def _set_controls_enabled(self, enabled: bool):
        if enabled:
            pass
        else:
            self.landscape_to_portrait_checkbox.setEnabled(False)
        for widget in [self.browse_button, self.api_key_edit, self.num_clips_spinbox,
                       self.clip_duration_spinbox, self.max_tasks_spinbox, self.start_button,
                       self.open_dir_button, self.language_combo, self.power_words_edit,
                       self.burn_subtitle_checkbox, self.precise_subtitle_checkbox,
                       self.add_bgm_checkbox, self.gpu_accel_checkbox,
                       self.whisper_model_combo, self.subtitle_lines_spinbox, self.max_chars_per_line_spinbox,
                       self.vcodec_combo, self.preset_combo, self.crf_spinbox, self.help_button]:
            if widget is not self.landscape_to_portrait_checkbox:
                widget.setEnabled(enabled)
        self.gpu_accel_checkbox.setEnabled(enabled and self.has_gpu)
        if enabled:
            self.update_vcodec_options()
            self.update_preset_options(self.vcodec_combo.currentText())
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
            'precise_subtitles': self.precise_subtitle_checkbox.isChecked(),
            'add_bgm': self.add_bgm_checkbox.isChecked(),
            'max_subtitle_lines': self.subtitle_lines_spinbox.value(),
            'max_chars_per_line': self.max_chars_per_line_spinbox.value(),
            'whisper_model': whisper_model_name,
            'vcodec': self.vcodec_combo.currentText(),
            'preset': self.preset_combo.currentText(),
            'crf': self.crf_spinbox.value(),
            'use_gpu': self.gpu_accel_checkbox.isChecked(),
            'convert_landscape_to_portrait': self.landscape_to_portrait_checkbox.isChecked()
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
                if self.input_path_edit.text():
                    self.check_video_orientation(self.input_path_edit.text())
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
        if self.input_path_edit.text():
            self.check_video_orientation(self.input_path_edit.text())
        QMessageBox.information(self, "完成", f"所有任务已处理完毕！\n片段已保存到 {self.output_dir_name} 文件夹。")
        self.thread, self.worker = None, None

    def on_error(self, error_message):
        self._set_controls_enabled(True)
        if self.input_path_edit.text():
            self.check_video_orientation(self.input_path_edit.text())
        self.progress_bar.setValue(0)
        QMessageBox.critical(self, "错误", f"处理过程中发生错误:\n{error_message}")
        self.thread, self.worker = None, None

    def closeEvent(self, event):
        self.save_config()
        if self.has_gpu:
            try:
                pynvml.nvmlShutdown()
                logger.info("pynvml 已成功关闭。")
            except pynvml.NVMLError:
                pass
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
        os.environ["PATH"] += os.pathsep + str(Path(sys.executable).parent)
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLESHEET)
    main_win = MainWindow()
    main_win.show()
    sys.exit(app.exec_())
