import os
from tempfile import NamedTemporaryFile


def save_uploaded_file(uploaded_file):
    with NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        return temp_file.name


def cleanup_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)
