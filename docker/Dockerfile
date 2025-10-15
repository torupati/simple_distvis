FROM python:3.12-slim

WORKDIR /app

COPY . /app

# Install uv if you want to use it, or just use pip
RUN pip install uv

# If you have uv.lock and pyproject.toml
# RUN uv pip sync --system --no-cache-dir uv.lock

# Or, if you use requirements.txt
#RUN uv pip freeze > requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
