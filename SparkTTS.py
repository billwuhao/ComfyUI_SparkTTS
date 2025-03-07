# Copyright (c) 2025 SparkAudio
#               2025 Xinsheng Wang (w.xinshawn@gmail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import re
import os
import torch
import numpy as np
from typing import Tuple
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

from sparktts.utils.file import load_config
from sparktts.models.audio_tokenizer import BiCodecTokenizer
from sparktts.utils.token_parser import LEVELS_MAP, GENDER_MAP, TASK_TOKEN_MAP


node_dir = os.path.dirname(os.path.abspath(__file__))
comfy_path = os.path.dirname(os.path.dirname(node_dir))
model_path = os.path.join(comfy_path, "models/TTS")
tts_model_path = os.path.join(model_path, "Spark-TTS-0.5B")
speaker_path = os.path.join(model_path, "Step-Audio-speakers")

class SparkTTS:
    """
    Spark-TTS for text-to-speech generation.
    """

    def __init__(self, model_dir: Path, device: torch.device = torch.device("cuda:0")):
        """
        Initializes the SparkTTS model with the provided configurations and device.

        Args:
            model_dir (Path): Directory containing the model and config files.
            device (torch.device): The device (CPU/GPU) to run the model on.
        """
        self.device = device
        self.model_dir = model_dir
        self.configs = load_config(f"{model_dir}/config.yaml")
        self.sample_rate = self.configs["sample_rate"]
        self._initialize_inference()

    def _initialize_inference(self):
        """Initializes the tokenizer, model, and audio tokenizer for inference."""
        self.tokenizer = AutoTokenizer.from_pretrained(f"{self.model_dir}/LLM")
        self.model = AutoModelForCausalLM.from_pretrained(f"{self.model_dir}/LLM")
        self.audio_tokenizer = BiCodecTokenizer(self.model_dir, device=self.device)
        self.model.to(self.device)

    def process_prompt(
        self,
        text: str,
        prompt_speech_path: Path,
        prompt_text: str = None,
    ) -> Tuple[str, torch.Tensor]:
        """
        Process input for voice cloning.

        Args:
            text (str): The text input to be converted to speech.
            prompt_speech_path (Path): Path to the audio file used as a prompt.
            prompt_text (str, optional): Transcript of the prompt audio.

        Return:
            Tuple[str, torch.Tensor]: Input prompt; global tokens
        """

        global_token_ids, semantic_token_ids = self.audio_tokenizer.tokenize(
            prompt_speech_path
        )
        global_tokens = "".join(
            [f"<|bicodec_global_{i}|>" for i in global_token_ids.squeeze()]
        )

        # Prepare the input tokens for the model
        if prompt_text is not None:
            semantic_tokens = "".join(
                [f"<|bicodec_semantic_{i}|>" for i in semantic_token_ids.squeeze()]
            )
            inputs = [
                TASK_TOKEN_MAP["tts"],
                "<|start_content|>",
                prompt_text,
                text,
                "<|end_content|>",
                "<|start_global_token|>",
                global_tokens,
                "<|end_global_token|>",
                "<|start_semantic_token|>",
                semantic_tokens,
            ]
        else:
            inputs = [
                TASK_TOKEN_MAP["tts"],
                "<|start_content|>",
                text,
                "<|end_content|>",
                "<|start_global_token|>",
                global_tokens,
                "<|end_global_token|>",
            ]

        inputs = "".join(inputs)

        return inputs, global_token_ids

    def process_prompt_control(
        self,
        gender: str,
        pitch: str,
        speed: str,
        text: str,
    ):
        """
        Process input for voice creation.

        Args:
            gender (str): female | male.
            pitch (str): very_low | low | moderate | high | very_high
            speed (str): very_low | low | moderate | high | very_high
            text (str): The text input to be converted to speech.

        Return:
            str: Input prompt
        """
        assert gender in GENDER_MAP.keys()
        assert pitch in LEVELS_MAP.keys()
        assert speed in LEVELS_MAP.keys()

        gender_id = GENDER_MAP[gender]
        pitch_level_id = LEVELS_MAP[pitch]
        speed_level_id = LEVELS_MAP[speed]

        pitch_label_tokens = f"<|pitch_label_{pitch_level_id}|>"
        speed_label_tokens = f"<|speed_label_{speed_level_id}|>"
        gender_tokens = f"<|gender_{gender_id}|>"

        attribte_tokens = "".join(
            [gender_tokens, pitch_label_tokens, speed_label_tokens]
        )

        control_tts_inputs = [
            TASK_TOKEN_MAP["controllable_tts"],
            "<|start_content|>",
            text,
            "<|end_content|>",
            "<|start_style_label|>",
            attribte_tokens,
            "<|end_style_label|>",
        ]

        return "".join(control_tts_inputs)

    @torch.no_grad()
    def inference(
        self,
        text: str,
        prompt_speech_path: Path = None,
        prompt_text: str = None,
        gender: str = None,
        pitch: str = None,
        speed: str = None,
        temperature: float = 0.8,
        top_k: float = 50,
        top_p: float = 0.95,
    ) -> torch.Tensor:
        """
        Performs inference to generate speech from text, incorporating prompt audio and/or text.

        Args:
            text (str): The text input to be converted to speech.
            prompt_speech_path (Path): Path to the audio file used as a prompt.
            prompt_text (str, optional): Transcript of the prompt audio.
            gender (str): female | male.
            pitch (str): very_low | low | moderate | high | very_high
            speed (str): very_low | low | moderate | high | very_high
            temperature (float, optional): Sampling temperature for controlling randomness. Default is 0.8.
            top_k (float, optional): Top-k sampling parameter. Default is 50.
            top_p (float, optional): Top-p (nucleus) sampling parameter. Default is 0.95.

        Returns:
            torch.Tensor: Generated waveform as a tensor.
        """
        if gender is not None:
            prompt = self.process_prompt_control(gender, pitch, speed, text)

        else:
            prompt, global_token_ids = self.process_prompt(
                text, prompt_speech_path, prompt_text
            )
        model_inputs = self.tokenizer([prompt], return_tensors="pt").to(self.device)

        # Generate speech using the model
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=3000,
            do_sample=True,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
        )

        # Trim the output tokens to remove the input tokens
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        # Decode the generated tokens into text
        predicts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # Extract semantic token IDs from the generated text
        pred_semantic_ids = (
            torch.tensor([int(token) for token in re.findall(r"bicodec_semantic_(\d+)", predicts)])
            .long()
            .unsqueeze(0)
        )

        if gender is not None:
            global_token_ids = (
                torch.tensor([int(token) for token in re.findall(r"bicodec_global_(\d+)", predicts)])
                .long()
                .unsqueeze(0)
                .unsqueeze(0)
            )

        # Convert semantic tokens back to waveform
        wav = self.audio_tokenizer.detokenize(
            global_token_ids.to(self.device).squeeze(0),
            pred_semantic_ids.to(self.device),
        )

        return wav


with open(f"{speaker_path}/speakers_info.json", "r", encoding="utf-8") as f:
    speakers_info = json.load(f)
speakers = list(speakers_info.keys())

class SparkTTSRun:
    @classmethod
    def INPUT_TYPES(s):
        
        return {
            "required": {
                "text": ("STRING", {"default": "", "multiline": True}),
                "speaker": (speakers, {"default": "婷婷"}),
                "gender": (["female", "male", "None"],{"default": "female"}),
                "pitch": (["very_low", "low", "moderate", "high", "very_high"],{"default": "moderate"}),
                "speed": (["very_low", "low", "moderate", "high", "very_high"],{"default": "moderate"}),
                # "temperature": ("FLOAT", {"default": 0.8, "min": 0, "max": 1, "step": 0.1}),
                # "top_k": ("INT", {"default": 50, "min": 0}),
                # "top_p": ("FLOAT", {"default": 0.95, "min": 0, "max": 1, "step": 0.01}),
                # "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "speak"
    CATEGORY = "MW-Spark-TTS"

    def speak(self, text, speaker, gender, pitch, speed, 
            #   temperature, 
            #   top_k, 
            #   top_p, 
            #   seed
              ):
        gender = None if gender == "None" else gender
        model = SparkTTS(tts_model_path)
        prompt_speech_path = f"{speaker_path}/{speaker}_prompt.wav"
        texts = [i.strip() for i in text.split("\n\n") if  i.strip()]
        audio_data = []
        for i in texts:
            with torch.no_grad():
                wav = model.inference(
                    i,
                    prompt_speech_path,
                    prompt_text=speakers_info[speaker],
                    gender=gender,
                    pitch=pitch,
                    speed=speed,
                    # top_k=top_k,
                    # top_p=top_p,
                    # temperature=temperature,
                )

            audio_data.append(wav)
        combined_wav = np.concatenate(audio_data)
            
        audio_tensor = torch.from_numpy(combined_wav).unsqueeze(0).unsqueeze(0).float()
        return ({"waveform": audio_tensor, "sample_rate": 16000},)


class SparkTTSClone:
    @classmethod
    def INPUT_TYPES(s):
        
        return {
            "required": {
                "text": ("STRING", {"default": "", "multiline": True}),
                # "gender": (["female", "male", "None"],{"default": "female"}),
                # "pitch": (["very_low", "low", "moderate", "high", "very_high"],{"default": "moderate"}),
                # "speed": (["very_low", "low", "moderate", "high", "very_high"],{"default": "moderate"}),
                "clone_text": ("STRING", {"default": "", "multiline": True, "tooltip": "The clone audio's text."}),
                "clone_audio": ("AUDIO", ),
                # "temperature": ("FLOAT", {"default": 0.8, "min": 0, "max": 1, "step": 0.1}),
                # "top_k": ("INT", {"default": 50, "min": 0}),
                # "top_p": ("FLOAT", {"default": 0.95, "min": 0, "max": 1, "step": 0.01}),
                # "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "clone"
    CATEGORY = "MW-Spark-TTS"

    def clone(self, text, clone_text, clone_audio, 
            #   gender, pitch, speed,
            #   temperature, 
            #   top_k, 
            #   top_p, 
            #   seed
              ):
        # gender = None if gender == "None" else gender
        
        prompt_wav = clone_audio["waveform"].squeeze(0).squeeze(0).numpy()
        prompt_wav_sr = clone_audio["sample_rate"]
        
        import soundfile as sf
        import io
        # 创建字节流缓冲区
        buffer = io.BytesIO()
        # 将音频数据写入 WAV 格式的字节流
        sf.write(buffer, prompt_wav, prompt_wav_sr, format='WAV')
        # 获取字节流数据
        audio_bytes = buffer.getvalue()
        # 关闭缓冲区（可选，BytesIO 不需要显式关闭，但为了清晰）
        buffer.close()
        # 创建新的 BytesIO 对象以读取字节流
        clone_audio_path = io.BytesIO(audio_bytes)
        model = SparkTTS(tts_model_path)

        texts = [i.strip() for i in text.split("\n\n") if  i.strip()]
        audio_data = []
        for i in texts:
            with torch.no_grad():
                wav = model.inference(
                    i,
                    clone_audio_path,
                    prompt_text=clone_text,
                    # gender=gender,
                    # pitch=pitch,
                    # speed=speed,
                    # top_k=top_k,
                    # top_p=top_p,
                    # temperature=temperature,
                )

            clone_audio_path.seek(0) 
            audio_data.append(wav)
        combined_wav = np.concatenate(audio_data)
            
        audio_tensor = torch.from_numpy(combined_wav).unsqueeze(0).unsqueeze(0).float()
        return ({"waveform": audio_tensor, "sample_rate": 16000},)

    
from AudioRecorderSpark import AudioRecorderSpark

NODE_CLASS_MAPPINGS = {
    "SparkTTSRun": SparkTTSRun,
    "SparkTTSClone": SparkTTSClone,
    "AudioRecorderSpark": AudioRecorderSpark
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SparkTTSRun": "Spark TTS Run",
    "SparkTTSClone": "Spark TTS Clone",
    "AudioRecorderSpark": "MW Audio Recorder for Spark"
}