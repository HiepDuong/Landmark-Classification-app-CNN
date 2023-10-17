# Landmark Classification App & API

This repository contains the FastAPI-based web API for landmark classification. The API allows users to upload an image and get back the top landmarks the image might be depicting, along with their respective probabilities.

## Prerequisites

- **Anaconda**: This project uses Anaconda for managing virtual environments and dependencies. Ensure you have Anaconda installed.
  
- **Python**: The codebase is written in Python and was tested with version 3.7.6. Please ensure you have a compatible Python version.

## Setup Using Anaconda
0. **Open anaconda promt**

1. **Create a New Anaconda Environment**:
conda create --name anyname python=3.7.6 pytorch=1.11.0 torchvision torchaudio cudatoolkit -c pytorch

3. **Activate the Environment**:
conda activate anyname

4. **Navigate to Project Directory**:
conda activate anyname
   
5. **Install Requirements**:
pip install -r requirements.txt

6. **Running API locally**
uvicorn deploy_app:app --reload

## How to Run

### 1. Using the App Interface:
   
- Open the `app.ipynb` notebook.
- Ensure you've installed all dependencies:

### 2. Using the `/predict/` Endpoint through Swagger UI:

**1. Open the Swagger UI in your browser:**  
Navigate to [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

**2. Locate the `/predict/` POST endpoint:**  
On the Swagger UI page, you'll see a list of available API endpoints on the left side. Find the `/predict/` endpoint listed under the `POST` method.

**3. Try out the API:**  
   - Click on the `/predict/` endpoint to expand its details.
   - Press the "Try it out" button.
   - You will see an option to upload a file. Select an image from your computer using this option.
   - Once your image is selected, hit the "Execute" button to send the request.

**4. Review the response:**  
After processing the request, the Swagger UI will show the server's response below the "Execute" button. This will include the predicted landmarks along with their probabilities.
