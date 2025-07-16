import streamlit as st
import Chatterbox
import torch
import random
import numpy as np
from scipy.io.wavfile import write

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_model():
    model = Chatterbox.ChatterboxTTS.from_pretrained(DEVICE)
    return model

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

def generate(model, text, audio_prompt_path, exaggeration, temperature, seed_num, cfgw):
    model = load_model() if model is None else model
    if seed_num != 0:
        set_seed(int(seed_num))

    wav = model.generate(
        text,
        audio_prompt_path=audio_prompt_path,
        exaggeration=exaggeration,
        temperature=temperature,
        cfg_weight=cfgw
    )
    return (model.sr, wav.squeeze(0).numpy())

st.title("TTS App")
text_input = st.text_area("Enter text here:")

if st.button("Generate Audio", type="primary", help="Click to generate audio from the text input"):
    with st.spinner("Generating audio..."):
        output_file = "output.wav"
        model = load_model()
        sr, waveform = generate(model=None,
            text=text_input,
            audio_prompt_path=None,  # Placeholder for audio prompt path
            exaggeration=0.8,  # Default value
            temperature=0.8,  # Default value
            cfgw=0.5,  # Default value
            seed_num=0  # Default value
        )
        write(output_file, sr, waveform)
        audio_file = open(output_file, "rb")
        st.audio(audio_file.read(), format="audio/wav")