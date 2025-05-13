# Iris Flower Prediction Model Deployment with Flask

This repository demonstrates how to deploy a machine learning model using **Flask** to predict the class of Iris flowers based on features like Sepal Length, Sepal Width, Petal Length, and Petal Width.

This simple web application serves as a beginner-friendly example of how to take an ML model and deploy it in a production-like setting using **Flask**. It also contains basic HTML forms to accept user input for predictions.

---

The repository contains the following files and directories:

- `app.py` - The main Flask application that runs the web server and handles requests.
- `model.ipynb` - Jupyter notebook where the machine learning model is trained using the Iris dataset.
- `templates/` - Contains HTML files (e.g., `index.html`) that provide the front-end for the application.
- `requirements.txt` - Lists all the necessary libraries for the project.
- `flask_beginners_guide.ipynb` - provides a step-by-step guide for Flask beginners.
- `html_beginners_guide.ipynb` - provides a basic guide on HTML, including form elements and other beginner concepts.

---

<img src="templates/Screenshot.png" alt="Iris APP Screenshot" width="500" height ="550" />


## How to Run
1. Clone this repository to your local machine:
   ```bash
   git clone https://github.com/your-username/Iris-Flower-Prediction-Model-Deployment.git
   cd Iris-Flower-Prediction-Model-Deployment
   ````

2. Create a virtual environment and activate it:

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Run the Flask application:

   ```bash
   python app.py
   ```

5. Open your browser and go to `http://127.0.0.1:5000/` to access the app and make predictions.

