from fastapi import FastAPI, File, UploadFile
import whisper
import os
import uuid

app = FastAPI()
model = whisper.load_model("small")  # Puedes usar tiny, base, small, medium, large

@app.post("/transcribe/")
async def transcribe_audio(file: UploadFile = File(...)):
    # Guardar archivo temporal
    temp_filename = f"temp_{uuid.uuid4()}.wav"
    with open(temp_filename, "wb") as buffer:
        buffer.write(await file.read())

    # Transcribir con Whisper
    result = model.transcribe(temp_filename, language="es",verbose=False, temperature=0.0)
    os.remove(temp_filename)
    return {"text": result["text"]}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
