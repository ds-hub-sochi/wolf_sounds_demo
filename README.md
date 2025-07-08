# Description

This is an official demo repository of "Automated detection of wolf howls using Audio Spectrogram Transformers" paper.
Here you can find a minimal example of models usage, as well as a proper weights downloading procedure.

# Usage

Online demo is available at the [link](demo-wolf.ds-hub.ru/). Please not that for some reasons online demo can be unavailable. In this case you can easily build and run demo locally. See next section to build proposed demo correctly.

Pre-trained weights will be downloaded automatically, but feel free to download them from the [yandex disk](https://disk.yandex.ru/client/disk/Wolf%20howl%20detection).
There you will find pre-trained weights for both models, as well as some data examples.

# Local build and run

First of all, you need to create an image vid:

```bash
docker build -t streamlit_demo .
```

The you need to run a container based on the image you've created:

```bash
docker run --gpus "0" -it --rm -p 8501:8501 streamlit_demo:latest
```

In the case you have more than 1 GPU you can pass another dvice id. Also you can remove '--gpu "0"' argument to infer the models using CPU only.
