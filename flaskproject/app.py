from flask import Flask, request, jsonify, send_from_directory, send_file
import os
from train import transcribe_audio  # Import the transcription function
from finetune import start_fine_tune
from flask_cors import CORS
import zipfile

app = Flask(__name__)
CORS(app)
# Configure the upload folder and allowed file extensions
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'wav'}  # Assuming you're uploading WAV files

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create the uploads folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    """Check if the uploaded file is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/fine_tune')
def home():
    result = start_fine_tune('audio')
    return jsonify(message=result), 201 

@app.route('/upload_audio', methods=['POST'])
def upload_file():
    # """Handle file upload."""
    if 'file' not in request.files:
        return jsonify(error="No file part"), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify(error="No selected file"), 400
    
    if file and allowed_file(file.filename):
        filename = file.filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        UPLOAD_FOLDER = transcribe_audio(file_path, "../new/model")  # Change this to your actual path
        app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
        zip_filename = 'files.zip'

        # Create a ZIP file containing the specified files
        with zipfile.ZipFile(zip_filename, 'w') as zipf:
            for file in os.listdir(UPLOAD_FOLDER) :  # List your actual file names here
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], file)
                if os.path.exists(file_path):
                    zipf.write(file_path, file)  # Add file to the ZIP
                else:
                    return jsonify({"error": f"{file} not found"}), 404
        
        return send_file(zip_filename, as_attachment=True)

    #     # Call the transcribe function
    #     # transcription = transcribe_audio(file_path)
    #     return jsonify(message="File uploaded successfully!", filename=filename), 201

        #return jsonify(message="File uploaded successfully!", filename=filename, transcription=transcription), 201
    
    # return jsonify(error="File type not allowed"), 400

@app.route('/upload', methods=['POST'])
def upload():
    # """Handle file upload."""
    if 'file' not in request.files:
        return jsonify(error="No file part"), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify(error="No selected file"), 400
    
    if file and allowed_file(file.filename):
        filename = file.filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        return jsonify(message="File uploaded successfully!", filename=filename), 201

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
