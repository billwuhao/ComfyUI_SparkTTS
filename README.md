[中文](README-CN.md) | [English](README.md)

# Spark-TTS ComfyUI Node

Using Spark-TTS in ComfyUI. Spark-TTS: An efficient text-to-speech model based on LLM with clone sounds from various languages.

![](https://github.com/billwuhao/ComfyUI_SparkTTS/blob/master/images/2025-03-07_03-08-47.png)

## Updates

[2025-03-21] ⚒️: Refactored code, optional model unloading, faster generation speed. Added more tunable parameters. Supports cross-lingual voice cloning.

[2025-03-07] ⚒️: Released version v1.0.0. New recording node `MW Audio Recorder for Spark` can be used to record audio with a microphone, and the progress bar displays the recording progress:

![](https://github.com/billwuhao/ComfyUI_StepAudioTTS/blob/master/assets/2025-03-06_21-29-09.png)

## Installation

```
cd ComfyUI/custom_nodes
git clone https://github.com/billwuhao/ComfyUI_SparkTTS.git
cd ComfyUI_SparkTTS
pip install -r requirements.txt

# python_embeded
./python_embeded/python.exe -m pip install -r requirements.txt
```

## Model Download

Download the following models to the `ComfyUI\models\TTS` folder.

[Spark-TTS-0.5B](https://huggingface.co/SparkAudio/Spark-TTS-0.5B)

Move the `Step-Audio-speakers` folder from this repository to the `ComfyUI\models\TTS` folder.

The structure should look like this:
```
ComfyUI\models\TTS
├── Spark-TTS-0.5B
├── Step-Audio-speakers
```

**Note**: If you have already installed [ComfyUI_StepAudioTTS](https://github.com/billwuhao/ComfyUI_StepAudioTTS), there’s no need to move it, as they share audio and configuration files.

You can then freely customize speakers under the `ComfyUI\models\TTS\Step-Audio-speakers` folder for use. Ensure that the speaker name configuration matches exactly:

![](https://github.com/billwuhao/ComfyUI_SparkTTS/blob/master/images/2025-03-07_03-30-51.png)

## Acknowledgments

[Spark-TTS](https://github.com/SparkAudio/Spark-TTS.git)