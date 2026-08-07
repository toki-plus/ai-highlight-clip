# AI Highlight Clip

将长视频转化为候选高光片段的 AI 辅助剪辑工作流。

项目结合语音识别、大模型内容评分、时间轴定位、视频裁剪和字幕生成，帮助课程、访谈和直播团队更快完成长内容的初步筛选。最终选片和发布决策仍由人工完成。

## 项目背景

从数小时视频中寻找适合二次分发的片段，通常需要人工完整观看、记录时间点、剪辑并重新制作标题和字幕。该流程耗时，并且不同编辑人员的筛选标准难以保持一致。

AI Highlight Clip 将初筛过程拆分为可观察的处理步骤，并保留关键词、目标时长和片段数量等人工配置项。

## 目标用户

- 课程、访谈、播客和直播内容团队
- 需要处理大量长视频素材的运营人员
- 研究 AI 辅助视频理解与剪辑流程的开发者

## 主要能力

- 使用 Whisper 将视频语音转为带时间轴的文本
- 根据提示词、关键词和上下文对候选片段进行评分
- 生成候选时间区间与标题
- 使用 FFmpeg 裁剪片段并合并字幕
- 支持单文件和目录批量输入
- 在 PyQt5 界面中展示任务进度和日志

## 工作流

```text
Long-form video
    -> Speech transcription
    -> Time-aligned transcript
    -> LLM-assisted scoring
    -> Candidate time ranges
    -> FFmpeg clipping
    -> Human review and export
```

## 技术设计

- `LLMClient`：模型调用和结构化结果处理
- `SubtitleService`：语音识别与字幕处理
- `AIPipeline`：评分、标题和剪辑流程编排
- `Processor`：媒体文件处理
- `MainWindow`：参数配置、任务状态和输出管理

主要技术：Python、OpenAI Whisper、DashScope、Pydantic、FFmpeg、PyQt5。

## 快速开始

### 环境要求

- Python 3.8+
- FFmpeg（需加入 `PATH`；Windows 仓库中包含兼容版本）
- DashScope API Key

```bash
git clone https://github.com/toki-plus/ai-highlight-clip.git
cd ai-highlight-clip
python -m venv venv
```

激活虚拟环境后：

```bash
pip install -r requirements.txt
python yanqu_highlight_clip.py
```

API Key 可在应用界面中配置。不要将真实密钥提交到仓库。

## 使用边界

- 模型评分只能作为初筛建议，不能替代人工内容判断。
- 输出质量受录音质量、语言、模型和提示词影响。
- 处理长视频时需要较多计算资源和磁盘空间。
- 当前版本尚未建立完整的自动化测试与 CI。

## 项目价值

本项目验证了“非结构化长视频 → 结构化文本 → 候选片段 → 人工确认”的端到端方案，而不是完全自动替代编辑人员。

## License

See [LICENSE](./LICENSE).
