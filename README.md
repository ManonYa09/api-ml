## Test With Expirement
### Training
- **EDA**: `Quick Exploration`, `Data Cleaning`, `Daignostics Analysis`,
- **Data Preprocessing**: Encoding, Feature Engineering
- **Model Training**: Seletion, Model, Parameter, algorithm
- **Model Evauation**: Evaution matric
### Automation Traning

## Install Visual Environment
### Mac User: 
- Create: python3 -m venv myven(name environment)
- Activate: source myven(name environment)/bin/activate
### Window User
- Create: python3 -m venv myven(name environment)
- Activate: myven(name environment)\Scripts\activate
### Install requrements.txt
- pip install -r requrements.txt


### If you get Port Already in use error while using mlflow
    - Get List of Services & PID running
    `sudo lsof -i tcp:5000`
    - Kill them 
    `kill -15 <PID>`