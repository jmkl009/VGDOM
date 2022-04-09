# VGDOM
Code Repository of VGDOM: Visual Grammar Modeling for Cross-Template Attribute Extraction
Datasets and implementation of FreeDOM and SimpDOM available at https://drive.google.com/drive/folders/14kdHMbXjCOsDXSWSjfCUctFOlqg_e54V?usp=sharing
### Installation
Python Version: 3.8

1. Install the packages
   ```sh
   conda install cudatoolkit=11.1.74 -c nvidia
   conda install pytorch=1.8.1 torchvision=0.9.1 -c pytorch
   conda install dgl-cuda11.1 -c dglteam 
   conda install pandas requests tqdm numba
   ```
2. To select the dataset being tested, modify the path to the directory and the class names in constants.py. 
3. Start your training and testing process
   ```sh
   python3 train_vgdom_main.py -e [epoch number] -cvf [fold number] -bs [batch_size]
   ```

<p align="right">(<a href="#top">back to top</a>)</p>

