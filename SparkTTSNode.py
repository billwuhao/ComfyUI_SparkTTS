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
import tempfile
import numpy as np
from typing import Tuple
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

from sparktts.utils.file import load_config
from sparktts.models.audio_tokenizer import BiCodecTokenizer
from sparktts.utils.token_parser import (LEVELS_MAP, 
                                        GENDER_MAP, 
                                        TASK_TOKEN_MAP, 
                                        # AGE_MAP, 
                                        # EMO_MAP
                                        )


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
        prompt_speech_path:  Path,
        prompt_text: str = None,
    ) -> Tuple[str, torch.Tensor]:
        """
        Process input for voice cloning.

        Args:
            text (str): The text input to be converted to speech.
            prompt_speech_path: Path to the audio file used as a prompt.
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
        # age, 
        # emotion,
        pitch: str,
        speed: str,
        text: str,
        # pitch_var, 
        # loudness,
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
        # age_id = AGE_MAP[age]
        # emotion_id = EMO_MAP[emotion]
        # pitch_var_level_id = LEVELS_MAP[pitch_var]
        # loudness_level_id = LEVELS_MAP[loudness]

        # pitch_var_tokens = f"<|pitch_var_label_{pitch_var_level_id}|>"
        # loudness_label_tokens = f"<|loudness_label_{loudness_level_id}|>"
        # age_tokens = f"<|age_{age_id}|>"
        # emotion_tokens = f"<|emotion_{emotion_id}|>"
        pitch_label_tokens = f"<|pitch_label_{pitch_level_id}|>"
        speed_label_tokens = f"<|speed_label_{speed_level_id}|>"
        gender_tokens = f"<|gender_{gender_id}|>"

        attribte_tokens = "".join(
            [gender_tokens, 
            pitch_label_tokens, 
            speed_label_tokens, 
            # age_tokens, 
            # emotion_tokens, 
            # pitch_var_tokens, 
            # loudness_label_tokens
            ]
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
        # age: str = None,
        # emotion: str = None,
        pitch: str = None,
        speed: str = None,
        # pitch_var = None, 
        # loudness = None,
        temperature: float = 0.8,
        top_k: float = 50,
        top_p: float = 0.95,
        max_new_tokens=3000,
    ) -> torch.Tensor:
        """
        Performs inference to generate speech from text, incorporating prompt audio and/or text.

        Args:
            text (str): The text input to be converted to speech.
            prompt_speech_path: Path to the audio file used as a prompt.
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
            prompt = self.process_prompt_control(gender, 
                                                #  age, 
                                                #  emotion, 
                                                 pitch, 
                                                 speed, 
                                                 text, 
                                                #  pitch_var, 
                                                #  loudness
                                                 )

        else:
            prompt, global_token_ids = self.process_prompt(
                text, prompt_speech_path, prompt_text
            )
        model_inputs = self.tokenizer([prompt], return_tensors="pt").to(self.device)

        # Generate speech using the model
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
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


class SparkTTSRun:
    @classmethod
    def INPUT_TYPES(s):
        
        return {
            "required": {
                "text": ("STRING", {"default": "", "multiline": True}),
                "gender": (["female", "male"],{"default": "female"}),
                # "age": (["Child", "Teenager", "Youth-Adult", "Middle-aged", "Elderly"], {"default": "Youth-Adult"}),
                # "emotion": (["UNKNOWN", "NEUTRAL", "ANGRY", "HAPPY", "SAD", "FEARFUL", "DISGUSTED", "SURPRISED", 
                #            "SARCASTIC", "EXCITED", "SLEEPY", "CONFUSED", "EMPHASIS", "LAUGHING", "SINGING", 
                #            "WORRIED", "WHISPER", "ANXIOUS", "NO-AGREEMENT", "APOLOGETIC", "CONCERNED", 
                #            "ENUNCIATED", "ASSERTIVE", "ENCOURAGING", "CONTEMPT"], {"default": "NEUTRAL"}),
                "pitch": (["very_low", "low", "moderate", "high", "very_high"],{"default": "moderate"}),
                "speed": (["very_low", "low", "moderate", "high", "very_high"],{"default": "moderate"}),
                # "pitch_var": (["very_low", "low", "moderate", "high", "very_high"],{"default": "moderate"}),
                # "loudness": (["very_low", "low", "moderate", "high", "very_high"],{"default": "moderate"}),
                # "temperature": ("FLOAT", {"default": 0.8, "min": 0, "max": 1, "step": 0.1}),
                # "top_k": ("INT", {"default": 50, "min": 0}),
                # "top_p": ("FLOAT", {"default": 0.95, "min": 0, "max": 1, "step": 0.01}),
                # "max_new_tokens": ("INT", {"default": 3000, "min": 500}),
                # "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "speak"
    CATEGORY = "MW-Spark-TTS"

    def speak(self, text, gender, 
            #   age, 
            #   emotion, 
              pitch, 
              speed, 
            #   pitch_var, 
            #   loudness,
            #   temperature, 
            #   top_k, 
            #   top_p, 
            #   max_new_tokens,
            #   seed
              ):

        model = SparkTTS(tts_model_path)

        texts = [i.strip() for i in text.split("\n\n") if  i.strip()]
        audio_data = []
        for i in texts:
            with torch.no_grad():
                wav = model.inference(
                    i,
                    gender=gender,
                    # age=age,
                    # emotion=emotion,
                    pitch=pitch,
                    speed=speed, 
                    # pitch_var=pitch_var, 
                    # loudness=loudness,
                    # top_k=top_k,
                    # top_p=top_p,
                    # temperature=temperature,
                    # max_new_tokens=max_new_tokens,
                )

            audio_data.append(wav)
        combined_wav = np.concatenate(audio_data)
            
        audio_tensor = torch.from_numpy(combined_wav).unsqueeze(0).unsqueeze(0).float()
        return ({"waveform": audio_tensor, "sample_rate": 16000},)


with open(f"{speaker_path}/speakers_info.json", "r", encoding="utf-8") as f:
    speakers_info = json.load(f)
speakers = list(speakers_info.keys())

class SparkTTSClone:
    @classmethod
    def INPUT_TYPES(s):
        
        return {
            "required": {
                "text": ("STRING", {"default": "", "multiline": True}),
                "cloned_speaker": (speakers, {"default": "婷婷", "tooltip": "Cloned speaker already defined in the JSON file. If you choose `custom_clone_audio`, it will be invalid"}),
                # "temperature": ("FLOAT", {"default": 0.8, "min": 0, "max": 1, "step": 0.1}),
                # "top_k": ("INT", {"default": 50, "min": 0}),
                # "top_p": ("FLOAT", {"default": 0.95, "min": 0, "max": 1, "step": 0.01}),
                # "max_new_tokens": ("INT", {"default": 3000, "min": 500}),
                # "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "custom_clone_text": ("STRING", {"default": "", "multiline": True, "tooltip": "(optional) The clone audio's text."}),
                "custom_clone_audio": ("AUDIO", ),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "clone"
    CATEGORY = "MW-Spark-TTS"

    def clone(self, text, cloned_speaker, custom_clone_text=None, custom_clone_audio=None, 
            #   temperature, 
            #   top_k, 
            #   top_p, 
            #   max_new_tokens,
            #   seed
              ):

        # 检查是否提供了自定义音频
        if custom_clone_audio is not None:
            # 检查是否提供了自定义文本
            if custom_clone_text and custom_clone_text.strip():
                clone_text = custom_clone_text
            else:
                clone_text = None
            # 提取传入的音频数据和采样率
            waveform = custom_clone_audio["waveform"].squeeze(0)
            sample_rate = custom_clone_audio["sample_rate"]
            
            import torchaudio

            # 创建临时文件
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                audio_file_path = temp_file.name
            
            # 使用 torchaudio.save() 保存临时文件
            torchaudio.save(audio_file_path, waveform, sample_rate, format="wav", 
                           bits_per_sample=16, encoding="PCM_S")

        else:
            audio_file_path = f"{speaker_path}/{cloned_speaker}_prompt.wav"
            clone_text = speakers_info[cloned_speaker]
        
        model = SparkTTS(tts_model_path)

        # 分割输入文本并生成音频
        texts = [i.strip() for i in text.split("\n\n") if i.strip()]
        audio_data = []
        for i in texts:
            with torch.no_grad():
                wav = model.inference(
                    i,
                    prompt_speech_path=audio_file_path,
                    prompt_text=clone_text,
                    gender=None,
                    # top_k=top_k,
                    # top_p=top_p,
                    # temperature=temperature,
                )
                audio_data.append(wav)

        # 生成完成后删除临时文件
        if custom_clone_audio is not None:
            if os.path.exists(audio_file_path):
                os.remove(audio_file_path)

        combined_wav = np.concatenate(audio_data)
            
        audio_tensor = torch.from_numpy(combined_wav).unsqueeze(0).unsqueeze(0).float()
        return ({"waveform": audio_tensor, "sample_rate": 16000},)

    
from MWAudioRecorderSpark import AudioRecorderSpark

NODE_CLASS_MAPPINGS = {
    "SparkTTSRun": SparkTTSRun,
    "SparkTTSClone": SparkTTSClone,
    "AudioRecorderSpark": AudioRecorderSpark
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SparkTTSRun": "Spark TTS Run",
    "SparkTTSClone": "Spark TTS Clone",
    "AudioRecorderSpark": "MW Audio Recorder"
}