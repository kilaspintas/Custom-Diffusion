# Custom Diffusion Studio (Intel IPEX Edition)

A simple and powerful Gradio interface application for generating images using custom Stable Diffusion (`.safetensors`) models. This application is designed and specifically optimized to run with **Intel® Extension for PyTorch (IPEX)** acceleration on Intel hardware.

## Key Features

* **Custom Model Loading**: Easily load your own Stable Diffusion 1.5-based `.safetensors` models.
* **Intuitive Interface**: A clean UI with a dropdown for model selection and sliders for all essential parameters.
* **IPEX Acceleration**: Optimized for fast inference on Intel GPUs (Arc/Iris Xe) using `bfloat16` and `ipex.optimize`.
* **Full Control**: Adjust Steps, Guidance Scale (CFG), Image Size (Width & Height), Seed, and Clip Skip.
* **Sampler Selection**: Choose from popular samplers like DPM++ 2M Karras, Euler a, and LCM for different artistic styles.
* **Automated Setup**: Comes with a `start.bat` script that automatically prepares the Conda environment and all necessary dependencies.

## System Requirements

This project is currently developed and tested specifically for the following environment:

* **Operating System**: Windows 10 / 11
* **GPU**: **Intel Arc™** or **Intel® Iris® Xe Graphics**. 
* **Required Software**: **Anaconda** or **Miniconda**.

## Installation Guide (First Time Only)

Follow these steps to set up the application for the first time.

1.  **Install Anaconda**:
    If you don't have Anaconda, download and install it from the official website. Use a version with Python 3.10 or newer.
    * **Download Link**: [**Anaconda Distribution**](https://www.anaconda.com/download/success)

2.  **Download the Project**:
    Download or clone this project.

3.  **Prepare Your Models**:
    * Create a new folder named `model` inside your project folder.
    * Place all your `.safetensors` model files (must be SD 1.5-based) inside this `model` folder.

4.  **Run the Automated Setup**:
    * **Double-click the `start.bat` file**.
    * A terminal window will appear and begin the setup process. This will take a considerable amount of time as it will create a new Conda environment and install all required libraries, including PyTorch and IPEX.
    * Let this process run until you see the message "SETUP FOR INTEL COMPLETED!".

## How to Run the Application

After the first-time installation is complete, for every subsequent use:

* Simply **double-click the `start.bat` file**.
* The script will automatically detect that the environment already exists, activate it, and launch the Gradio interface.

## Project Folder Structure
```
YOUR_PROJECT_FOLDER/
│
├── app.py # Contains the user interface code (Gradio)
├── backend
	└── backend_core.py # Contains all core logic (model loading, inference)
├── environment.yml # The "recipe" for standard dependencies in the Conda environment
├── start.bat # The main script for setup and running the application
│
└── model/ # Folder to place all your .safetensors files
	├── custom_model_1.safetensors
	└── custom_model_2.safetensors
```
