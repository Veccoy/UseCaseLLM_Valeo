FROM python:3.10.14-alpine3.20

RUN groupadd --gid 1024 docker
RUN adduser --disable-password --uid 1055 llmuser

WORKDIR /home/lluser

COPY ../requirements.txt /home/lluser/
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
RUN pip install --no-cache-dir --upgrade -r /home/llmuser/requirements.txt

COPY ../app /home/lluser/
RUN chown -R vbercy /home/llmuser

RUN rm -Rf /root/.cache/pip && \
    rm -rf /tmp/*

USER llmuser

