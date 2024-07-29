## Описание

Этот репозиторий содержит демку для выявления воя волком на аудио-фалах. Скорее всего, формат решения временный и при развитии проекта, связанного с аудио, оно будет меняться, но пока решили оставить демку в таком виде.

## Установка и запуск

Демку предполагается запускать в докере, поэтому прежде всего нужно собрать образ.

```bash
docker build -t streamlit_demo .
```

Далее можно на основе созданного образа запустить контейнер.

```bash
docker run --gpus "0" -it -v $(pwd):/app -p 8501:8501 streamlit_demo:latest
```

N.B. если мы всё же решим на использовать GPU, нужно просто убрать '--gpus "0"' из команды.