from project_name.models.audio_feature_svm import AudioFeatureSVM

from typing import Annotated
import os

from fastapi import (
    FastAPI,
    File,
    UploadFile,
    HTTPException,
    Form,
    Request,)
from fastapi.responses import HTMLResponse

app = FastAPI()

STYLE = """
    body {
        font-family: Arial, sans-serif;
        background: #f4f4f4;
        margin: 0;
        padding: 0;
    }
    .container {
        max-width: 600px;
        margin: 40px auto;
        background: #fff;
        padding: 30px 40px;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    .select_feature {
        margin-top: 40px;
        text-align: center;
    }
    .upload-form, .model-selection {
        max-width: 600px;
        margin: 40px auto;
        background: #fff;
        padding: 30px 40px;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        text-align: center;
    }
    .model-selection {
        margin-top: 40px;
    }
    .page-links {
        text-align: center;
        margin-top: 20px;
    }
    .predict {
        margin-top: 40px;
    }
    h2 {
        color: #333;
        margin-bottom: 20px;
    }
    p {
        color: #333;
        margin-bottom: 20px;
    }
    h1 {
        color: #333;
        margin-bottom: 20px;
    }
    ul {
        list-style-type: disc;
        padding-left: 20px;
    }
    li {
        margin-bottom: 8px;
        color: #444;
    }
    .btn {
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
    }
    .btn:hover {
        background: #0056b3;
    }
"""

@app.exception_handler(HTTPException)
async def custom_http_exception_handler(request: Request, exc: HTTPException):
    content = f"""
    <html>
    <head>
        <title>Error</title>
        <style>
            {STYLE}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Error {exc.status_code}</h1>
            <p>{exc.detail}</p>
            <a href="/" class="btn">Back to Home</a>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=content, status_code=exc.status_code)

@app.get("/")
async def main():
    model_options = os.listdir("project_name/saved_models")
    model_options = [f.split(".")[0] for f in model_options if f.endswith(".joblib")]
    content = f"""
    <style>
        {STYLE}
    </style>
    <body>
    <div class="upload-form">
        <h1>Upload multiple files</h1>
        <form action="/uploadfiles/" enctype="multipart/form-data" method="post">
        <input name="files" type="file" multiple>
        <input type="submit">
        </form>
        <p>Upload .wav files only.</p>
        <p>Files will be saved in the 'uploadedfiles' directory.</p>
    </div>
    <div class="model-selection">
        <h1>Select Model</h1>
        <form action="/select_model/" method="post">
            <select name="model", type="text">
                <option value="" disabled selected>Select a model</option>
                {''.join(f'<option value="{model}">{model}</option>' for model in model_options)}
            </select>
            <input type="submit">
        </form>
    </div>
    <div class="select-pages">
        <a href="/uploadedfiles/", class="btn">View Uploaded Files</a>
    </div>
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
    <style>
        {STYLE}
    </style>
    <body>
        <div class="container">
            <h1>Uploaded Files</h1>
            <ul>
                files: {filenames}
            </ul>
        </div>
        <div class="page-links">
            <a href="/uploadedfiles/" class="btn">View Uploaded Files</a>
            <a href="/" class="btn">Back to home</a>
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
    <style>
        {STYLE}
    </style>
    <div class="container">
        <h1>Uploaded Files</h1>
        {file_list}
    </div>
    <div class="page-links">
        <a href="/" class="btn">Back to home</a>
    </div>
    </body>
    """
    return HTMLResponse(content=content)

# User select model from home to use
@app.post("/select_model/", response_class=HTMLResponse)
async def select_model(model: str = Form(...)):
    audio_files = os.listdir("uploadedfiles")
    if os.path.exists(f"project_name/saved_models/{model}.joblib"):
        # Here you can implement logic to set the selected model
        content = f"""
        <style>
            {STYLE}
        </style>
        <body>
            <div class="container">
                <h1>Model {model} selected successfully!</h1>
            </div>
            <div class="select_feature">
                <h1>Audio Selections</h1>
                <form action="/feature_selection/" method="post">
                    <input type="hidden" name="model" value="{model}">
                    <label for="audio_files">Select Audio Files:</label>
                    <select name="audio_files" multiple>
                        <option value="" disabled>Select audio files</option>
                        {''.join(f'<option value="{fname}">{fname}</option>' for fname in audio_files)}
                    </select>
                    <input type="submit" value="Select Features" class="btn">
                </form>
            <div class="page-links">
                <a href="/" class="btn">Back to home</a>
                <a href="/uploadedfiles/" class="btn">View Uploaded Files</a>
            </div>
        </body>
        """
        return HTMLResponse(content=content)
    else:
        raise HTTPException(
            status_code=404,
            detail=f"Model {model} not found in saved models. Please train the model first."
        )

@app.post("/feature_selection/", response_class=HTMLResponse)
async def feature_selection(
    model: str = Form(...),
    audio_files: list[str] = Form(...),
):
    if not os.path.exists(f"project_name/saved_models/{model}.joblib"):
        raise HTTPException(
            status_code=404,
            detail=f"Model {model} not found. Please train the model first."
        )
    for audio_file in audio_files:
        if not os.path.exists(f"uploadedfiles/{audio_file}"):
            raise HTTPException(
                status_code=404,
                detail=f"Audio file {audio_file} not found."
            )

    # Here you would implement the feature selection logic
    content = f"""
    <body>
    <style>
        {STYLE}
    </style>
    <div class="container">
        <h1>Feature Selection Completed</h1>
        <ul>
            <li>Model: {model}</li>
            <li>Audio Files: {', '.join(audio_files)}</li>
        </ul>
    </div>
    <div class="predict">
        <h1>Make a Prediction</h1>
        <form action="/predict/" method="post">
            <input type="hidden" name="model" value="{model}">
            <input type="hidden" name="audio_file" value="{audio_files}">
            <input type="submit" value="Predict" class="btn">
        </form>
    </div>
    <div class="page-links">
        <a href="/" class="btn">Back to Home</a>
        <a href="/uploadedfiles/" class="btn">View Uploaded Files</a>
    </div>
    </body>
    """
    # For now, we just return a success message
    return HTMLResponse(content=content)

@app.post("/predict/", response_class=HTMLResponse)
async def predict(
    model: str = Form(...),
    audio_files: list[str] = Form(...),
):
    if not os.path.exists(f"project_name/saved_models/{model}.joblib"):
        raise HTTPException(
            status_code=404,
            detail=f"Model {model} not found. Please train the model first."
        )
    
    for audio_file in audio_files:
        if not os.path.exists(f"uploadedfiles/{audio_file}"):
            raise HTTPException(
                status_code=404,
                detail=f"Audio file {audio_file} not found."
            )
    
    path_to_audios = os.path.join(os.getcwd(), "uploadedfiles")
    
    
    
    # Load the model
    match model:
        case "audio_svm":
            svm_model = AudioFeatureSVM.load(f"project_name/saved_models/{model}.joblib")
        case _:
            raise HTTPException(
                status_code=404,
                detail=f"Model {model} is not supported."
            )
    pre_processor = svm_model.preprocessor
    
    # Here you would implement the prediction logic
    content = f"""
    <body>
    <style>
        {STYLE}
    </style>
    <div class="container">
        <h1>Prediction Completed</h1>
        <p>Prediction for <strong>{audio_file}</strong> using model <strong>{model}</strong> has been successfully completed.</p>
        <a href="/" class="btn">Back to Home</a>
    </div>
    </body>
    """
    # For now, we just return a success message
    return HTMLResponse(content=content)
