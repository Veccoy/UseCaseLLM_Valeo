FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel

RUN groupadd docker
RUN adduser --disabled-password --home /home/llmuser --ingroup docker llmuser

WORKDIR /home/llmuser

COPY ./requirements.txt /home/llmuser/
RUN pip install --no-cache-dir --upgrade -r /home/llmuser/requirements.txt

COPY ./app/ /home/llmuser/
RUN chown -R llmuser /home/llmuser

RUN rm -Rf /root/.cache/pip && \
    rm -rf /tmp/*

USER llmuser