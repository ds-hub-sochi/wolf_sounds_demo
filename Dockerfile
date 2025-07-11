FROM pytorch/pytorch:2.2.1-cuda11.8-cudnn8-runtime

WORKDIR /app

RUN apt-get -y update
RUN apt-get -y upgrade
RUN apt-get install -y ffmpeg

COPY ./requirements.txt ./install/requirements.txt
COPY ./setup.py ./setup.py
RUN pip install -r ./install/requirements.txt

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

COPY ./streamlit_app.py ./streamlit_app.py
COPY ./src ./src

CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
