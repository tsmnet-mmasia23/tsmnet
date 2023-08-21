import os
import time
from tsmnet import Stretcher
import gradio as gr
from gradio import processing_utils
import torch
import numpy as np
import torchaudio
import yt_dlp
import noisereduce as nr

model_root = './weights'
yt_dl_dir = 'yt-audio'
available_models = ['speech', 'pop-music', 'classical-music']
working_sr = 22050

def prepare_models():
    return {
        weight: Stretcher(os.path.join(model_root, f'{weight}.pt'))
        for weight in available_models
    }

def download_yt_audio(url):
    # purge outdated audio files (older than 1 days)
    os.system(f'find {yt_dl_dir} -audio -mtime +1 -delete')

    ydl_opts = {
        'format': 'm4a/bestaudio/best',
        'postprocessors': [{  # Extract audio using ffmpeg
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
        }],
        'outtmpl': f"{yt_dl_dir}/%(id)s.%(ext)s"
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            ydl.cache.remove()
            meta = ydl.extract_info(url, download=False)
            audio_file = os.path.join(yt_dl_dir, meta['id'] + '.wav')
            if not os.path.isfile(audio_file):
                ydl.download(url)

        except yt_dlp.DownloadError as error:
            raise gr.Error(f'Failed to download from YouTube: {error}')

    new_audio_file = os.path.join(os.path.dirname(audio_file), f'{time.time()}.wav')
    os.system(f'cp {audio_file} {new_audio_file}')
    return new_audio_file


def prepare_audio_file(rec, audio_file, yt_url):
    if rec is not None:
        return rec
    if audio_file is not None:
        return audio_file
    if yt_url != '':
        return download_yt_audio(yt_url)
    else:
        raise gr.Error('No audio found!')


def run(rec, audio_file, yt_url, denoise, speed, model, start_time, end_time):
    audio_file = prepare_audio_file(rec, audio_file, yt_url)

    x, sr = torchaudio.load(audio_file)
    x = torchaudio.transforms.Resample(orig_freq=sr, new_freq=working_sr)(x)
    sr = working_sr

    x = x[:, int(start_time * sr):int(end_time * sr)]

    if speed == 1:
        torchaudio.save(audio_file, x, sr)
        return processing_utils.audio_from_file(audio_file)

    x = models[model](x, speed).cpu()

    if denoise:
        if len(x.shape) == 1: # mono
            x = x[None]
        x = x.numpy()
        # perform noise reduction
        x = torch.from_numpy(np.stack([nr.reduce_noise(y=y, sr=sr) for y in x]))

    torchaudio.save(audio_file, x, sr)
    return processing_utils.audio_from_file(audio_file)


# @@@@@@@ Start of the program @@@@@@@@

models = prepare_models()
os.makedirs(yt_dl_dir, exist_ok=True)

with gr.Blocks() as demo:
    gr.Markdown('# TSM-Net')
    gr.Markdown('---')
    with gr.Row():
        with gr.Column():
            with gr.Tab('From microphone'):
                rec_box = gr.Audio(label='Recording', source='microphone', type='filepath')
            with gr.Tab('From YouTube'):
                yt_url_box  = gr.Textbox(label='YouTube URL', placeholder='https://youtu.be/q6EoRBvdVPQ')
            with gr.Tab('From file'):
                audio_file_box = gr.Audio(label='Audio sample', type='filepath')
            denoise_box = gr.Checkbox(label='Speech enhancement (should be off for music)', value=True)

            rec_box.change(lambda: [None, None, True], outputs=[audio_file_box, yt_url_box, denoise_box])
            audio_file_box.change(lambda: [None, None, False], outputs=[rec_box, yt_url_box, denoise_box])
            yt_url_box.input(lambda: [None, None, False], outputs=[rec_box, audio_file_box, denoise_box])

            speed_box = gr.Slider(label='Playback speed', minimum=0.25, maximum=2, value=1)
            with gr.Accordion('Fine-grained settings', open=False):
                with gr.Tab('Trim audio sample (sec)'):
                    # gr.Markdown('### Trim audio sample (sec)')
                    with gr.Row():
                        start_time_box = gr.Number(label='Start', value=0)
                        end_time_box = gr.Number(label='End', value=60)
                model_box = gr.Dropdown(label='Model weight', choices=available_models, value=available_models[0])

            submit_btn = gr.Button('Submit')

        with gr.Column():
            with gr.Accordion('Hint', open=False):
                gr.Markdown('You can find more settings under the **Fine-grained settings**')
                gr.Markdown('- Waiting too long? Try to adjust the start/end timestamp')
                gr.Markdown('- Low audio quality? Try to switch to a proper model weight')
            outputs=gr.Audio(label='Output')

        submit_btn.click(fn=run, inputs=[
            rec_box,
            audio_file_box,
            yt_url_box,
            denoise_box,
            speed_box,
            model_box,
            start_time_box,
            end_time_box,
        ], outputs=outputs)

    with gr.Accordion('Read more ...', open=False):
        gr.Markdown('---')
        gr.Markdown(
            'We proposed a novel approach in the field of time-scale modification '
            'on audio signals. While traditional methods use the framing technique, '
            'spectral approach uses the short-time Fourier transform to preserve '
            'the frequency during temporal stretching. TSM-Net, our neural-network '
            'model encodes the raw audio into a high-level latent representation. '
            'We call it Neuralgram, in which one vector represents 1024 audio samples. '
            'It is inspired by the framing technique but addresses the clipping '
            'artifacts. The Neuralgram is a two-dimensional matrix with real values, '
            'we can apply some existing image resizing techniques on the Neuralgram '
            'and decode it using our neural decoder to obtain the time-scaled audio. '
            'Both the encoder and decoder are trained with GANs, which shows fair '
            'generalization ability on the scaled Neuralgrams. Our method yields '
            'little artifacts and opens a new possibility in the research of modern '
            'time-scale modification. Please find more detail in our '
            '<a href="https://arxiv.org/abs/2210.17152" target="_blank">paper</a>.'
        )

demo.queue(4)
demo.launch(server_name='0.0.0.0')

