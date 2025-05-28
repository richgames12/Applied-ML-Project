from typing import Annotated
import os

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse

app = FastAPI()

@app.get("/")
async def main():
    content = """
<body>
<h1>Upload multiple files</h1>
<form action="/uploadfiles/" enctype="multipart/form-data" method="post">
<input name="files" type="file" multiple>
<input type="submit">
</form>
<p>Upload .wav files only.</p>
<p>Files will be saved in the 'uploadedfiles' directory.</p>
</body>
"""
    return HTMLResponse(content=content)


@app.post("/uploadfiles/")
async def create_upload_files(
    files: Annotated[
        list[UploadFile], File(description="Multiple files as UploadFile")
    ],
):
    for file in files:
        if file.filename[-4:] != ".wav":
            raise HTTPException(
                status_code=400,
                detail=f"File {file.filename} is not a .wav file.",
            )
        # Here you can save the file or process it as needed
        # Ensure the directory exists
        if not os.path.exists("uploadedfiles"):
            os.makedirs("uploadedfiles")
        # Save the file
        if os.path.exists(f"uploadedfiles/{file.filename}"):
            raise HTTPException(
                status_code=400,
                detail=f"File {file.filename} already exists.",
            )
        with open(f"uploadedfiles/{file.filename}", "wb") as f:
            content = await file.read()
            f.write(content)
    return {"filenames": [file.filename for file in files]}