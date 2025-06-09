import os
from pathlib import Path
from typing import Annotated, Optional, Union
from uuid import uuid4

from fastapi import (
    FastAPI,
    File,
    UploadFile,
    HTTPException,
    Form,
    Request,
)
from fastapi.responses import HTMLResponse

from main import EMOTION_LABELS, INTENSITY_LABELS
from project_name.data.data_preprocessing import AudioPreprocessor
from project_name.features.audio_feature_extractor import AudioFeatureExtractor
from project_name.models.audio_feature_svm import AudioFeatureSVM


# This is where the audio uploaded by the user will be stored.
UPLOAD_DIR = Path("tmp/user_audio")
MODEL_DIR = Path("project_name/saved_models")

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
        text-align: center;
    }
    .select_feature {
        margin-top: 40px;
        text-align: center;
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


# This thing runs before and after every request to manage the session ID.
# It is used to separate uploaded files by user.
@app.middleware("http")
async def assign_session_id(request: Request, call_next):
    """
    Is called before and after every reqyest to manage the session ID. This id
    is used to separate uploaded files by user.
    """
    session_id = request.cookies.get("session_id")
    if not session_id:
        # Generate a new UUID for the session (just random series of letters)
        session_id = str(uuid4())

    # Store the session ID on the request state
    request.state.session_id = session_id

    response = await call_next(request)  # Here, the actual request is handled.

    # Set the session_id as an HTTP-only cookie in the response.
    response.set_cookie("session_id", session_id, httponly=True)
    return response


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
    try:
        # Find all models
        model_options = [f.stem for f in MODEL_DIR.glob("*.joblib")]
    except FileNotFoundError:
        model_options = []
        # If the directory does not exist, we can create it
        warning = """
        <h1>Warning</h1>
        <p>The 'project_name/saved_models' directory does not exist. Please train a model first.</p>
        <p>This can be done by running the main.py script.</p>
        <p>Once a model is trained, it will appear as an option for select model.</p>
        """
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
    <div class="page-links">
        <a href="/uploadedfiles/", class="btn">View Uploaded Files</a>
    </div>
    </body>
"""
    if "warning" in locals():
        content += f"""
        <div class="container">
            {warning}
        </div>
        """
    return HTMLResponse(content=content)


@app.post("/uploadfiles/")
async def create_upload_files(
    request: Request,
    files: Annotated[
        list[UploadFile], File(description="Multiple files as UploadFile")
    ],
):
    """
    Upload multiple .wav files.

    Validates and saves .wav files to the 'uploadedfiles' directory.
    Rejects non-.wav files or duplicates.
    """

    # The middleware will have been called first and connected a session id
    # to the current state. This id is used to create user specific folders for
    # uploads by the user.
    session_id = request.state.session_id

    # Ensure the directory exists
    session_upload_dir = UPLOAD_DIR / session_id
    session_upload_dir.mkdir(parents=True, exist_ok=True)

    filenames = []
    for file in files:
        if file.filename[-4:] != ".wav":
            raise HTTPException(
                status_code=400,
                detail=f"File {file.filename} is not a .wav file.",
            )
        file_path = session_upload_dir / file.filename
        if file_path.exists():
            raise HTTPException(
                status_code=400,
                detail=f"File {file.filename} already exists.",
            )
        # Save the file
        with open(file_path, "wb") as f:
            f.write(await file.read())
        filenames.append(file.filename)

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
async def list_uploaded_files(request: Request):
    """
    List uploaded audio files.

    Displays all files currently stored in the 'uploadedfiles' directory.
    """

    session_id = request.state.session_id
    session_upload_dir = UPLOAD_DIR / session_id

    files = []
    if session_upload_dir.exists():
        files = os.listdir(session_upload_dir)
    file_list = (
        "<ul>" + "".join(f"<li>{fname}</li>" for fname in files) + "</ul>"
        if files
        else "<p>No files uploaded.</p>"
    )
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
async def select_model(request: Request, model: Optional[str] = Form(None)):
    """
    Select a trained model for prediction.

    Checks if the selected model exists and prompts for selecting .wav files.
    """
    if not model:
        raise HTTPException(
            status_code=400, detail="No model selected. Please select a model."
        )

    session_id = request.state.session_id
    session_upload_dir = UPLOAD_DIR / session_id
    # Ensure the directory exists
    session_upload_dir.mkdir(parents=True, exist_ok=True)

    # There can be 0 files
    audio_files = os.listdir(session_upload_dir)
    model_path = MODEL_DIR / f"{model}.joblib"
    if model_path.exists():
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
                    <input type="submit" value="Select Files" class="btn">
                </form>
            </div>
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
            detail=f"Model {model} not found in saved models. Please train the model first.",
        )


@app.post("/feature_selection/", response_class=HTMLResponse)
async def feature_selection(
    request: Request,
    model: str = Form(...),
    audio_files: Optional[list[str]] = Form(None),
):
    """
    Perform feature selection on selected audio files using a model.

    Validates files and prepares data for prediction.
    """

    session_id = request.state.session_id
    session_upload_dir = UPLOAD_DIR / session_id

    if not audio_files:
        raise HTTPException(
            status_code=400, detail="No audio files selected for feature selection."
        )

    model_path = MODEL_DIR / f"{model}.joblib"
    if not model_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Model {model} not found. Please train the model first.",
        )
    for audio_file in audio_files:
        if not (session_upload_dir / audio_file).exists():
            raise HTTPException(
                status_code=404, detail=f"Audio file {audio_file} not found."
            )

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
            {''.join(f'<input type="hidden" name="audio_files" value="{fname}">' for fname in audio_files)}
            <input type="submit" value="Predict" class="btn">
        </form>
    </div>
    <div class="page-links">
        <a href="/" class="btn">Back to Home</a>
        <a href="/uploadedfiles/" class="btn">View Uploaded Files</a>
    </div>
    </body>
    """
    return HTMLResponse(content=content)


@app.post("/predict/", response_class=HTMLResponse)
async def predict(
    request: Request,
    model: str = Form(...),
    audio_files: Union[list[str], str] = Form(...),
):
    """
    Predict emotion or intensity from uploaded audio files.

    Loads the selected model and extracted features to run predictions.
    """

    session_id = request.state.session_id
    session_upload_dir = UPLOAD_DIR / session_id

    # Normalize audio_files to a list
    if isinstance(audio_files, str):
        audio_files = [audio_files]

    model_path = MODEL_DIR / f"{model}.joblib"
    if not model_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Model {model} not found. Please train the model first.",
        )

    for audio_file in audio_files:
        if not (session_upload_dir / audio_file).exists():
            raise HTTPException(
                status_code=404, detail=f"Audio file {audio_file} not found."
            )

    match model:
        case "intensity_svm":
            selected_model = AudioFeatureSVM.load(model_path)
            pre_processor = AudioPreprocessor(
                sampling_rate=22050,
                target_length=66150,
                use_spectrograms=False,
                n_mels=128,
                n_fft=2048,
                hop_length=512,
            )
        case _:
            raise HTTPException(
                status_code=404, detail=f"Model {model} is not supported."
            )

    # Collect the absolute paths to the audios
    all_paths = []
    path_to_audios = session_upload_dir.resolve()

    for audio_file in audio_files:
        full_file_path = path_to_audios / audio_file
        if not full_file_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Audio file {audio_file} not found in uploaded files.",
            )
        # Preprocessing requires strings, not path objects
        all_paths.extend([str(full_file_path)])

    processed_audios, _, _ = pre_processor.process_all(all_paths)
    if len(processed_audios) == 0:
        raise HTTPException(
            status_code=400,
            detail="No valid audio files found for prediction.",
        )

    # Extract features before prediction
    feature_extractor = AudioFeatureExtractor(use_deltas=True, n_mfcc=20)
    features = feature_extractor.extract_features_all(processed_audios)

    predictions = selected_model.predict(features)

    if len(predictions) == 0:
        raise HTTPException(
            status_code=400,
            detail="No predictions could be made. Check the audio files and model.",
        )
    match model:
        case "intensity_svm":
            predictions = [INTENSITY_LABELS[pred - 1] for pred in predictions]
        case "emotion_svm":
            predictions = [EMOTION_LABELS[pred - 1] for pred in predictions]
        case _:
            raise HTTPException(
                status_code=404, detail=f"Model {model} is not supported."
            )

    content = f"""
    <body>
    <style>
        {STYLE}
    </style>
    <div class="container">
        <h1>Prediction Completed</h1>
        <p>Prediction for selected audio using model <strong>{model}</strong> has been successfully completed.</p>
        <ul>
            <li>Model: {model}</li>
            <li>Audio Files: {', '.join(audio_files)}</li>
            <li>Predictions: {', '.join(str(pred) for pred in predictions)}</li>
        </ul>
    </div>
    <div class="page-links">
        <a href="/uploadedfiles/" class="btn">View Uploaded Files</a>
        <a href="/" class="btn">Back to Home</a>
    </div>
    </body>
    """
    return HTMLResponse(content=content)
