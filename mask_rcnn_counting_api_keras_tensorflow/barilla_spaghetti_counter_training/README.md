## Requirements
Python 3.4, TensorFlow 1.3, Keras 2.0.8 and other common packages listed in `requirements.txt`.

## Installation
1. Install dependencies
   ```bash
   pip3 install -r requirements.txt
   ```
2. Clone this repository
3. Run setup from the repository root directory
    ```bash
    python3 setup.py install
    ``` 
4. Install `pycocotools` from one of these repos. Here are listed the necessary commands:
    ```bash
    pip3 install Cython
    git clone https://github.com/pdollar/coco.git  
    cd coco/PythonAPI
    make
    sudo make install
    sudo python3 setup.py install
    ``` 
## Run the code
Change the line#12 of main.py, set the input image name and then just run the command: python3 main.py then the result image will be saved as "result.png".
