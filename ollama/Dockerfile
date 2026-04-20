FROM ollama/ollama:latest AS downloader
RUN ollama serve & sleep 5 && ollama pull qwen2.5:7b

FROM ollama/ollama:latest
COPY --from=downloader /root/.ollama /root/.ollama
