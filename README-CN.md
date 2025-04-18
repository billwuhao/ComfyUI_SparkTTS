[中文](README-CN.md) | [English](README.md)

# Spark-TTS 的 Comfyui 节点

在 Comfyui 中使用 Spark-TTS. Spark-TTS: 一种基于 LLM 的高效文本到语音模型，能克隆各种语言的声音.

![](https://github.com/billwuhao/ComfyUI_SparkTTS/blob/master/images/2025-03-07_03-08-47.png)


## 更新

[2025-03-21]⚒️: 重构代码, 可选是否卸载模型, 生成速度更快. 添加更多可调参数. 支持克隆不同语言之间的声音.

[2025-03-07]⚒️: 发布版本 v1.0.0. 录音节点 `MW Audio Recorder for Spark` 可用麦克风录制音频, 进度条显示录制进度:

![](https://github.com/billwuhao/ComfyUI_StepAudioTTS/blob/master/assets/2025-03-06_21-29-09.png)

## 安装

```
cd ComfyUI/custom_nodes
git clone https://github.com/billwuhao/ComfyUI_SparkTTS.git
cd ComfyUI_SparkTTS
pip install -r requirements.txt

# python_embeded
./python_embeded/python.exe -m pip install -r requirements.txt
```

## 模型下载

下列模型下载到 `ComfyUI\models\TTS` 文件夹中.

[Spark-TTS-0.5B](https://https://huggingface.co/SparkAudio/Spark-TTS-0.5B)

将本仓库中 `Step-Audio-speakers` 文件夹移动到 `ComfyUI\models\TTS` 文件夹中. 

结构如下:

```
ComfyUI\models\TTS
├── Spark-TTS-0.5B
├── Step-Audio-speakers
```

**注意**: 如果你已经安装过 [ComfyUI_StepAudioTTS](https://github.com/billwuhao/ComfyUI_StepAudioTTS), 则无需移动, 它们是共享音频和配置文件的.

然后就可在 `ComfyUI\models\TTS\Step-Audio-speakers` 文件夹下随意自定义说话者即可使用. 注意说话者名称配置一定要一致:

![](https://github.com/billwuhao/ComfyUI_SparkTTS/blob/master/images/2025-03-07_03-30-51.png)

## 致谢

[Spark-TTS](https://github.com/SparkAudio/Spark-TTS.git)