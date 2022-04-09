# VGDOM
Code Repository of VGDOM: Visual Grammar Modeling for Cross-Template Attribute Extraction
Datasets and implementation of FreeDOM and SimpDOM available at https://drive.google.com/drive/folders/14kdHMbXjCOsDXSWSjfCUctFOlqg_e54V?usp=sharing
### Installation

1. Install the packages
   ```sh
   pip install -r requirements.txt
   ```
2. To select the dataset being tested, modify the path to the directory and the class names in constants.py. 
3. Start your training and testing process
   ```sh
   python3 train_vgdom_main.py -e [epoch number] -cvf [fold number] -bs [batch_size]
   ```

<p align="right">(<a href="#top">back to top</a>)</p>

