import json
import re
import os
import torch
import tempfile
import numpy as np
from typing import Tuple
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys
import logging
import platform
import gc
import folder_paths

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from sparktts.utils.file import load_config
from sparktts.models.audio_tokenizer import BiCodecTokenizer
from sparktts.utils.token_parser import (TokenParser,
                                        LEVELS_MAP, 
                                        GENDER_MAP, 
                                        TASK_TOKEN_MAP, 
                                        AGE_MAP, 
                                        EMO_MAP
                                        )


models_dir = folder_paths.models_dir
model_path = os.path.join(models_dir, "TTS")
tts_model_path = os.path.join(model_path, "Spark-TTS-0.5B")
speaker_path = os.path.join(model_path, "Step-Audio-speakers")

# Convert device argument to torch.device
if platform.system() == "Darwin" and torch.backends.mps.is_available():
    # macOS with MPS support (Apple Silicon)
    device = torch.device("mps")
    logging.info(f"Using MPS device: {device}")
elif torch.cuda.is_available():
    # System with CUDA support
    device = torch.device("cuda")
    logging.info(f"Using CUDA device: {device}")
else:
    # Fall back to CPU
    device = torch.device("cpu")
    logging.info("GPU acceleration not available, using CPU")


def load_models(device):
    tokenizer = AutoTokenizer.from_pretrained(f"{tts_model_path}/LLM")
    model = AutoModelForCausalLM.from_pretrained(f"{tts_model_path}/LLM")
    model.to(device).eval()
    audio_tokenizer = BiCodecTokenizer(tts_model_path, device=device)

    return tokenizer, model, audio_tokenizer

class SparkTTS:
    """
    Spark-TTS for text-to-speech generation.
    """

    def __init__(self,
                tokenizer, model, audio_tokenizer, 
                device: torch.device = torch.device("cuda:0")):
        """
        Initializes the SparkTTS model with the provided configurations and device.

        Args:
            model_dir (Path): Directory containing the model and config files.
            device (torch.device): The device (CPU/GPU) to run the model on.
        """
        self.tokenizer = tokenizer
        self.model = model
        self.audio_tokenizer = audio_tokenizer
        self.device = device
        self.configs = load_config(f"{tts_model_path}/config.yaml")
        self.sample_rate = self.configs["sample_rate"]

    def cleanup(self):
        self.tokenizer = None
        self.model = None
        self.audio_tokenizer = None
        gc.collect()
        torch.cuda.empty_cache()

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
        # assert pitch in range(1001)
        # assert speed in range(11)
        assert pitch in LEVELS_MAP.keys()
        assert speed in LEVELS_MAP.keys()

        gender_id = GENDER_MAP[gender]
        pitch_level_id = LEVELS_MAP[pitch]
        speed_level_id = LEVELS_MAP[speed]

        # pitch_value_id = pitch
        # speed_value_id = speed
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

        # pitch_value_tokens = f"<|pitch_value_{pitch_value_id}|>"
        # speed_value_tokens = f"<|speed_value_{speed_value_id}|>"

        gender_tokens = f"<|gender_{gender_id}|>"

        attribte_tokens = "".join(
            [gender_tokens, 
            pitch_label_tokens, 
            speed_label_tokens, 

            # pitch_value_tokens,
            # speed_value_tokens,

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
        do_sample: bool = True,
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
            do_sample=do_sample,
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
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.audio_tokenizer = None
        self.device = device

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
                # "pitch": ("INT",{"default": "500", "min": 0, "max": 1000, "step": 1}),
                # "speed": ("INT",{"default": "5", "min": 0, "max": 10, "step": 1}),
                # "pitch_var": (["very_low", "low", "moderate", "high", "very_high"],{"default": "moderate"}),
                # "loudness": (["very_low", "low", "moderate", "high", "very_high"],{"default": "moderate"}),
                "temperature": ("FLOAT", {"default": 0.8, "min": 0, "max": 1, "step": 0.1}),
                "top_k": ("INT", {"default": 50, "min": 0}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0, "max": 1, "step": 0.01}),
                "max_new_tokens": ("INT", {"default": 3000, "min": 500}),
                "do_sample": ("BOOLEAN", {"default": True}),
                "unload_model": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "speak"
    CATEGORY = "🎤MW/MW-Spark-TTS"

    def speak(self, text, gender, 
            #   age, 
            #   emotion, 
              pitch, 
              speed, 
              unload_model,
            #   pitch_var, 
            #   loudness,
              temperature, 
              top_k, 
              top_p, 
              max_new_tokens,
              do_sample,
              seed,
              ):
        if seed != 0:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        if self.model is None:
            self.tokenizer, self.model, self.audio_tokenizer = load_models(self.device)

        tts_model = SparkTTS(self.tokenizer, self.model, self.audio_tokenizer, self.device)

        wav = tts_model.inference(
                                text,
                                gender=gender,
                                # age=age,
                                # emotion=emotion,
                                pitch=pitch,
                                speed=speed, 
                                # pitch_var=pitch_var, 
                                # loudness=loudness,
                                top_k=top_k,
                                top_p=top_p,
                                temperature=temperature,
                                max_new_tokens=max_new_tokens,
                                do_sample=do_sample,
        )
        audio_tensor = torch.from_numpy(wav).unsqueeze(0).unsqueeze(0).float()

        if unload_model:
            tts_model.cleanup()
            self.model = None
            self.tokenizer = None
            self.audio_tokenizer = None
            gc.collect()
            torch.cuda.empty_cache()
            
        return ({"waveform": audio_tensor, "sample_rate": 16000},)


with open(f"{speaker_path}/speakers_info.json", "r", encoding="utf-8") as f:
    speakers_info = json.load(f)
speakers = list(speakers_info.keys())

class SparkTTSClone:
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.audio_tokenizer = None
        self.device = device

    @classmethod
    def INPUT_TYPES(s):
        
        return {
            "required": {
                "text": ("STRING", {"default": "", "multiline": True}),
                "cloned_speaker": (speakers, {"default": "婷婷", "tooltip": "Cloned speaker already defined in the JSON file. If you choose `custom_clone_audio`, it will be invalid"}),
                "temperature": ("FLOAT", {"default": 0.8, "min": 0, "max": 1, "step": 0.1}),
                "top_k": ("INT", {"default": 50, "min": 0}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0, "max": 1, "step": 0.01}),
                "max_new_tokens": ("INT", {"default": 3000, "min": 500}),
                "do_sample": ("BOOLEAN", {"default": True}),
                "unload_model": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "custom_clone_text": ("STRING", {"default": "", "multiline": True, "tooltip": "(optional) The clone audio's text."}),
                "custom_clone_audio": ("AUDIO", ),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "clone"
    CATEGORY = "🎤MW/MW-Spark-TTS"

    def clone(self, text, 
              cloned_speaker,
              temperature, 
              top_k, 
              top_p, 
              max_new_tokens,
              do_sample,
              unload_model,
              seed,
              custom_clone_text=None, 
              custom_clone_audio=None, 
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
        
        if seed != 0:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        if self.model is None:
            self.tokenizer, self.model, self.audio_tokenizer = load_models(self.device)

        tts_model = SparkTTS(self.tokenizer, self.model, self.audio_tokenizer, self.device)

        wav = tts_model.inference(
                                text,
                                prompt_speech_path=audio_file_path,
                                prompt_text=clone_text,
                                gender=None,
                                top_k=top_k,
                                top_p=top_p,
                                temperature=temperature,
                                max_new_tokens=max_new_tokens,
                                do_sample=do_sample,
                            )
        audio_tensor = torch.from_numpy(wav).unsqueeze(0).unsqueeze(0).float()

        if unload_model:
            tts_model.cleanup()
            self.model = None
            self.tokenizer = None
            self.audio_tokenizer = None
            gc.collect()
            torch.cuda.empty_cache()

        # 生成完成后删除临时文件
        if custom_clone_audio is not None:
            if os.path.exists(audio_file_path):
                os.remove(audio_file_path)

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