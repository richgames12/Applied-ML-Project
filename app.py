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
<a href="/uploadedfiles/">View Uploaded Files</a>
</body>
"""
    return HTMLResponse(content=content)


@app.post("/uploadfiles/")
async def create_upload_files(
    files: Annotated[
        list[UploadFile], File(description="Multiple files as UploadFile")
    ],
):
    filenames = [file.filename for file in files]
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
    content = f"""
    <html>
    <head>
        <title>Uploaded Files</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                background: #f4f4f4;
                margin: 0;
                padding: 0;
            }}
            .container {{
                max-width: 600px;
                margin: 40px auto;
                background: #fff;
                padding: 30px 40px;
                border-radius: 8px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            }}
            h1 {{
                color: #333;
                margin-bottom: 20px;
            }}
            ul {{
                list-style-type: disc;
                padding-left: 20px;
            }}
            li {{
                margin-bottom: 8px;
                color: #444;
            }}
            .btn {{
                display: inline-block;
                margin-top: 20px;
                padding: 10px 20px;
                background: #007bff;
                color: #fff;
                border: none;
                border-radius: 4px;
                text-decoration: none;
                font-size: 16px;
                cursor: pointer;
                transition: background 0.2s;
            }}
            .btn:hover {{
                background: #0056b3;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Uploaded Files</h1>
            <ul>
                files: {filenames}
            </ul>
            <a href="/" class="btn">Back to Upload</a>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=content)

@app.get("/uploadedfiles/", response_class=HTMLResponse)
async def list_uploaded_files():
    files = []
    if os.path.exists("uploadedfiles"):
        files = os.listdir("uploadedfiles")
    file_list = "<ul>" + "".join(f"<li>{fname}</li>" for fname in files) + "</ul>" if files else "<p>No files uploaded.</p>"
    content = f"""
    <body>
    <h1>Uploaded Files</h1>
    {file_list}
    <a href="/">Back to upload</a>
    </body>
    """
    return HTMLResponse(content=content)

