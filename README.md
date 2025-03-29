
1. **Install Python 3.8.10**:
   - Run the python installer provided in the repositry. 

2. **Install Visual Studio Code**:
   - Download and install VS Code from the [official website](https://code.visualstudio.com/).

3. **Install the Python Extension for VS Code**:
   - Open VS Code.
   - Go to the Extensions view by clicking on the Extensions icon in the Activity Bar on the side of the window.
   - Search for "Python" and install the extension provided by Microsoft.

4. **Clone the Project Repository**:
   - Open a terminal or command prompt.
   - Navigate to the directory where you want to clone the repository.
   - Run the following command to clone the repository:
     ```
     git clone https://github.com/mayureshagashe2105/HackDSC2021-Drowsiness-Detection-Web-app.git
     ```

5. **Navigate to the Project Directory**:
   - Change into the project directory:
  

6. ** Activate the virtual environment:**
     - On Windows:
       ```
       .venv\Scripts\activate
       ```
     - On macOS or Linux:
       ```
       source .venv/bin/activate
       ```

7. **Install Required Dependencies**: [Already installed in Environment run if not working]
   - Ensure that the `requirements.txt` file is present in the project directory. This file should contain:
     ```
     opencv_python==4.5.1.48
     tensorflow_gpu==2.3.1
     numpy==1.18.5
     pandas==1.1.4
     streamlit==0.73.1
     tensorflow==2.4.1
     ```
   - Install the dependencies using pip:
     ```
     pip install -r requirements.txt
     ```

8. **Download the Required Dataset**:
   - The project requires a dataset for drowsiness detection, which can be downloaded from [Kaggle](https://www.kaggle.com/kutaykutlu/drowsiness-detection).
   - Extract the dataset and place it in the appropriate directory as expected by the project's code.

9. **Run the Application**:
   - Launch the application using Streamlit:
     ```
     streamlit run Eye_patch_extractor_&_GUI.py
     ```
   - This should start the web application, and you can access it through the URL provided in the terminal.

**Note**: Ensure that your system has the necessary hardware and drivers to support TensorFlow GPU if you plan to use GPU acceleration. If not, you may need to adjust the TensorFlow installation to use the CPU version.

For the env file DM me on Instagram or Discord as its huge file doesnt allows me to upload the file without gitLTS so I just zipped it. Enjoyy!!!
