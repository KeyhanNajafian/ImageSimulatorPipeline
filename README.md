# Image Synthesis Pipeline
- - -
This pipeline includes all the needed tools for synthesizing images using the cut-and-paste approach in overlapping wheat spikes on the background images.
<p align="center">
  <img src="ImageSimulation.png" alt="Image Synthesis Pipeline" width="800" height="600">
</p>


### Code configuration and run:
To implement the code we used Python version 3.9, and a version of the packages can be installed through installing all the required packages.

1. In order to install the required packages please run the following command using `pip`: 
   ```bash
      pip install -r requirements.txt
   ```
> In order to run each python script you first need to change the related configuration file in the `config/` folder for each file.

2. In order to extract frames from videos:
   1. Change the configuration file named **frames_extractor.yaml** in the **configs/** folder.  
   2. Run the following command in the terminal:
   ```bash
      python3 frames_extractor.py --config configs/frames_extractor.yaml
   ```       
3. In order to extract real and fake objects from the segmented representative image:
   1. Change the configuration file named **objects_extractor.yaml** in the **configs/** folder. 
   2. Run the following command in the terminal:
   ```bash
      python3 objects_extractor.py --config configs/objects_extractor.yaml
   ```  
4. In order to simulate the dataset using the previously extracted background frames and real and fake objects:
   1. Change the configuration file named **simulation.yaml** in the **configs/** folder. 
   2. Run the following command in the terminal:
   ```bash
      python3 simulation.py --config configs/simulation.yaml
   ```  
### 📝 Cite
```
@article{najafian2022semi,
  title={Semi-Self-Supervised Learning for Semantic Segmentation in Images with Dense Patterns},
  author={Najafian, Keyhan and Ghanbari, Alireza and Kish, Mahdi Sabet and Eramian, Mark and Shirdel, Gholam Hassan and 
          Stavness, Ian and Jin, Lingling and Maleki, Farhad},
  journal={bioRxiv},
  year={2022},
  publisher={Cold Spring Harbor Laboratory}
}
```
