FROM python:3.12

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

EXPOSE 8080

ENV PORT=8080

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]