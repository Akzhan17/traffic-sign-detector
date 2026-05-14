#FROM ubuntu:latest
#LABEL authors="akzhanoskelen"

#ENTRYPOINT ["top", "-b"]
# Frontend Dockerfile
FROM python:3.10-slim

WORKDIR /app

RUN pip install streamlit requests pillow opencv-python numpy

COPY streamlit_app.py .

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]