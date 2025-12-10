# AI Highlight Clip: AI驱动的长视频高光时刻自动剪辑工具

[简体中文](./README.md) | [English](./README_en.md)

[![GitHub stars](https://img.shields.io/github/stars/toki-plus/ai-highlight-clip?style=social)](https://github.com/toki-plus/ai-highlight-clip/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/toki-plus/ai-highlight-clip?style=social)](https://github.com/toki-plus/ai-highlight-clip/network/members)
[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/toki-plus/ai-highlight-clip/pulls)

**`AI Highlight Clip` 是一款免费、开源的桌面应用程序，它能全自动地从长视频（如访谈、课程、直播回放）中，智能发现并剪辑出多个具有爆款潜质的“高光时刻”短视频。**

你是否曾面对数小时的视频素材，为了找到其中几个精彩片段而反复拉动进度条？本项目专为解决这一痛点而生，致力于将知识型、访谈型内容创作者从繁琐的剪辑工作中解放出来，实现内容生产力的指数级提升。

<p align="center">
  <a href="https://www.bilibili.com" target="_blank">
    <img src="./demo.png" alt="点击观看B站演示视频（暂未录制）" width="800"/>
  </a>
  <br>
  <em>(点击图片跳转到 B 站观看高清演示视频)</em>
</p>

---

## ✨ 核心功能

这不仅仅是一个剪辑工具，而是一个完整的AI内容再创作流水线：

-   **🤖 AI 驱动的核心引擎**:
    -   **自动语音识别**: 集成强大的 `OpenAI-Whisper` 模型，可精准识别多种语言，将视频中的对话自动转录为带时间戳的字幕文件。
    -   **AI 智能分析与评分**: 利用大语言模型（LLM）深度理解转录后的文本内容，为每个潜在的视频片段进行“高光指数”打分，自动发现最具价值和传播潜力的内容。
    -   **AI 爆款标题生成**: 模拟顶级内容营销专家，为筛选出的每个高光片段自动生成引人注目的、符合平台调性的短视频标题。

-   **🎬 全自动剪辑工作流**:
    -   **长视频到短视频矩阵**: 支持处理单个视频文件或整个视频目录（如一部连续剧、一套课程），一键将长内容转化为数十个可直接发布的短视频矩阵。
    -   **智能滑动窗口切片**: 通过高效的滑动窗口算法，无遗漏地扫描全部内容，确保不错过任何一个潜在的精彩瞬间。
    -   **高光片段智能筛选**: 根据AI评分、关键词密度和内容重叠度，自动筛选出TOP N个最佳片段，避免内容重复。

-   **🎨 高度可定制化输出**:
    -   **动态字幕嵌入**: 可选择将字幕自动嵌入到视频中，并支持自定义字体。内置高级算法，能优化字幕的断句和排版，提升观看体验。
    -   **参数自由配置**: 用户可自由设定期望的片段数量、目标时长、高光关键词等，灵活控制最终产出。
    -   **跨平台图形界面**: 基于 PyQt5 构建，提供在 Windows、macOS 和 Linux 上一致的、简洁直观的操作体验。

## 📸 软件截图

<p align="center">
  <img src="./images/cover_software.png" alt="软件主界面" width="800"/>
  <br>
  <em>升级版软件主界面。</em>
</p>
<p align="center">
  <img src="./images/cover_demo.png" alt="软件主界面" width="800"/>
  <br>
  <em>一键将 30 集 4K 电视剧《繁花》（44GB） 剪辑成 30 个金融相关的高光片段🎬</em>
</p>

## 🚀 快速开始

### 系统要求

1.  **Python**: 3.8 或更高版本。
2.  **FFmpeg**: **必须**安装 FFmpeg 并将其添加到系统环境变量中。
    -   Windows 用户：项目已内置 `ffmpeg.exe`，通常无需额外安装。
    -   macOS/Linux 用户：请访问 [FFmpeg 官网](https://ffmpeg.org/download.html) 查看安装教程。
    -   检查是否安装成功：打开终端或命令提示符，输入 `ffmpeg -version`，如果能看到版本信息则表示安装成功。
3.  **API Key**: 需要一个通义千问（DashScope）的 API Key。

### 安装与启动

1.  **克隆本仓库：**
    ```bash
    git clone https://github.com/toki-plus/ai-highlight-clip.git
    cd ai-highlight-clip
    ```

2.  **创建并激活虚拟环境 (推荐)：**
    ```bash
    python -m venv venv
    # Windows 系统
    venv\Scripts\activate
    # macOS/Linux 系统
    source venv/bin/activate
    ```

3.  **安装依赖库：**
    ```bash
    pip install -r requirements.txt
    ```
    *注意：`openai-whisper` 的安装可能需要 `rust` 编译环境。如果遇到问题，请参考其官方文档进行安装。*

4.  **配置 API Key:**
    -   首次运行前，打开 `config.ini` 文件。
    -   将你的通义千问 API Key 填入 `api_key` 字段。

5.  **运行程序：**
    ```bash
    python ai_highlight_clip.py
    ```

## 📖 使用指南

1.  **第一步：配置**
    -   在软件界面的 "千问（通义）API Key" 输入框中，粘贴你的 Key。
    -   根据需要选择 "语音识别语言" 和 "语音识别模型"（模型越精准，速度越慢，对硬件要求越高）。

2.  **第二步：选择输入**
    -   点击 "浏览..." 按钮，选择要处理的单个视频文件或包含多个视频的文件夹。

3.  **第三步：设定剪辑参数**
    -   **生成片段数量**：你希望最终得到多少个短视频。
    -   **目标片段时长**：每个短视频的大致时长（秒）。
    -   **高光关键词**：输入一些你认为重要的关键词（用逗号或空格隔开），AI会优先选择包含这些词的片段。
    -   **添加字幕**：勾选此项，最终的视频将自动带上字幕。

4.  **第四步：开始生成**
    -   点击 "🚀 开始生成" 按钮。
    -   程序将开始执行语音识别、AI分析、剪辑等一系列任务，你可以在日志窗口看到实时进度。
    -   处理时间取决于视频时长和你的电脑性能，请耐心等待。

5.  **第五步：获取成果**
    -   任务完成后，程序会弹出提示。
    -   点击 "📂 打开片段目录" 按钮，即可在 `output_clips` 文件夹中找到所有生成的短视频。

---

<p align="center">
  <strong>技术交流，请添加：</strong>
</p>
<table align="center">
  <tr>
    <td align="center">
      <img src="./images/wechat.png" alt="微信二维码" width="200"/>
      <br />
      <sub><b>个人微信</b></sub>
      <br />
      <sub>微信号: toki-plus (请备注“GitHub 定制”)</sub>
    </td>
    <td align="center">
      <img src="./images/gzh.png" alt="公众号二维码" width="200"/>
      <br />
      <sub><b>公众号</b></sub>
      <br />
      <sub>获取最新技术分享与项目更新</sub>
    </td>
  </tr>
</table>

## 📂 我的其他开源项目

-   **[AI Mixed-Cut](https://github.com/toki-plus/ai-mixed-cut)**: 一款颠覆性的AI内容生产工具，通过“解构-重构”模式将爆款视频解构成创作素材库，并全自动生成全新原创视频。
-   **[AI Video Workflow](https://github.com/toki-plus/ai-video-workflow)**: 全自动AI原生视频生成工作流，集成了文生图、图生视频和文生音乐模型，一键创作AIGC短视频。
-   **[AI TTV Workflow](https://github.com/toki-plus/ai-ttv-workflow)**: 一款AI驱动的文本转视频工具，能将任意文案自动转化为带有配音、字幕和封面的短视频，支持AI文案提取、二创和翻译。
-   **[Video Mover](https://github.com/toki-plus/video-mover)**: 一个强大的、全自动化的内容创作流水线工具。它可以自动监听、下载指定的博主发布的视频，进行深度、多维度的视频去重处理，并利用AI大模型生成爆款标题，最终自动发布到不同平台。
-   **[AB Video Deduplicator](https://github.com/toki-plus/AB-Video-Deduplicator)**: 通过创新的“高帧率抽帧混合”技术，从根本上重构视频数据指纹，以规避主流短视频平台的原创度检测和查重机制。

## 🤝 参与贡献

欢迎任何形式的贡献！如果你有新的功能点子、发现了Bug，或者有任何改进建议，请：
-   提交一个 [Issue](https://github.com/toki-plus/ai-highlight-clip/issues) 进行讨论。
-   Fork 本仓库并提交 [Pull Request](https://github.com/toki-plus/ai-highlight-clip/pulls)。

如果这个项目对你有帮助，请不吝点亮一颗 ⭐！

## 📜 开源协议

本项目基于 MIT 协议开源。详情请见 [LICENSE](LICENSE) 文件。
