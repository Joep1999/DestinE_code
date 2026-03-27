#!/bin/bash

cd ~/ResearchDataLab/floodMIND/running_rt


git clone https://github.com/Joep1999/DestinE_code.git
#cd DestinE_code/blending_code
# Create a virtual environment
python3 -m venv /nobackup_1/users/whan/floodmind/floodmind_rt_env
source /nobackup_1/users/whan/floodmind/floodmind_rt_env/bin/activate  
pip install -r ./DestinE_code/blending_code/requirements_light.txt
pip install --upgrade pip
## adding it for ipynb
# Press Ctrl+Shift+P
# Type: "Python: Select Interpreter"
# Click "Enter interpreter path..."
# Navigate to your environment's Python executable:
# /usr/people/whan/ResearchDataLab/floodMIND/DestinE_code/optimization_algorithm/floodmind_rt_env/bin/python


pip install optuna
pip install scoringrules
pip install "setuptools<81"
pip install ecmwf-api-client


datadir="/nobackup_1/users/whan/floodmind/floodmind_rt/"
mkdir -p $datadir
mkdir -p $datadir/input_general
mkdir -p $datadir/input_general/knmi_radar_gauge_adj
mkdir -p $datadir/input_general/knmi_radar
mkdir -p $datadir/input_general/p111_ecmwf_destine


## polytope stuff
#https://github.com/destination-earth-digital-twins/polytope-examples

git clone git@github.com:destination-earth-digital-twins/polytope-examples.git
pip install --upgrade polytope-client
pip install conflator
pip install lxml
pip install pyfftw
pip install hydra-core

cd ~/ResearchDataLab/floodMIND/polytope-examples
python desp-authentication.py -u  kiriwhan
#Password: [U...!1]
#Token successfully written to /usr/people/whan/.polytopeapirc

# also need ecmwf api set up...

# need file: knmi_grid.txt in /nobackup_1/users/whan/floodmind/floodmind_rt/input_general/knmi_radar_gauge_adj/
cp DestinE_code_v0/optimization_algorithm/knmi_data/knmi_grid.txt /nobackup_1/users/whan/floodmind/floodmind_rt/input_general/knmi_radar_gauge_adj/