# dalek_iris
Repo for Rubic Pi 3 co-processor for translating fuel from a 2d image to 3d (or rather, a 2d but from the top down rather than the side)
<br>Gives a birds eye view of the field

## Examples


## Install/Setup
``` bash
pip install -r requirements.txt
```
Calibrate the camera using this instructions: https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html

Put the images in calibrate/images/

Source and run 
``` bash 
source .venv/bin/activate
python calibrate/calib.py
```

Copy the output into ./config.json

## Running

Source and Run
``` bash
source .venv/bin/activate
python main.py
```

## Development

1) Updating requirements.txt
``` bash
pip > requirements.txt 
```

