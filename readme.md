<div align="center">
<h1>Hockey Tracking Dataset - 2 🏒 </h1>
</div>


This repository contains a curated public dataset for player tracking in ice Hockey. 

The associated publication is: 
[VIP-HTD : A Public Benchmark for Multi-Player Tracking in Ice Hockey](https://www.researchgate.net/publication/378204310_VIP-HTD_A_Public_Benchmark_for_Multi-Player_Tracking_in_Ice_Hockey)
### Abstract: 
Multi-Object Tracking (MOT) is the combined task of localization andassociation of subjects across a temporal sequence. Unlike the pop-ular pedestrian tracking paradigms, monocular tracking of ice hockeyplayers from broadcast feeds presents a variety of challenges dueto rapid non-linear motions, occlusions, blurs, and pan-tilt-zoom ef-fects. To tackle these issues, there neither exists public datasets norbenchmarks trained on public datasets to date. To this end, we propose: (a) HTD -2 (also called VIP-HTD) --  an open-sourced ice hockey tracking dataset, processed & curated from existing work (MHPTD), and (b) a public benchmark for multi-player tracking based on it. Further, we also present our observationsprocessing this dataset and discuss the two key metrics (IDF1 scoreand ID switches) required for optimal tracking evaluations. With thiswork, we take a step towards creating a uniﬁed public benchmark forevaluating multi-player tracking in hockey.

### Contents 

#### 1. Annotations
We have two major annoation formats: 
1. **Personnel-annotations:** This format assigns unique-player IDs throughout the entire sequence i.e., even if a player exits and re-enters the field-of-view (FoV) after a certain interval, he's given the same player ID. 
2. **MOT-challenge annotations:** This format assigns unique-player IDs only until a player exists within the FoV i.e., if he exits and re-enters back, he's given a new unique ID. 

We discuss in-depth about these two formats in the associated paper. 


#### 2. Frames

All 22 clips in this dataset can be found in the ./clips directory, their corresponding parse frames in the img1 folder. We use the [CVAT](https://github.com/opencv/cvat) tool to parse all clips at their native FPS rate. 
The utilities script contains basic pre-processing functions written in python to utilize this dataset. 



### Folder format:
```text
./clips
|
./mot-challenge-format
| --- test 
|  | --- CAR_VS_BOS_001 
|  |  | --- gt
|  |  |  | --- gt.txt 
|  |  | --- img1
|  |  |  | --- 000001.jpg
|  |  |  | --- 000002.jpg
|  |  |  | --- 000003.jpg
|  |  |  | --- 
|  | --- CAR_VS_NYR__001
|  | ---
|  |      
| --- train
|  | --- CAR_VS_BOS_002
|  | --- CAR_VS_NYR_002
|  | --- 
|  | ---
|
| --- validation 
|  | --- PIT_VS_WAS_001
|
./personnel-level-format
| --- test 
|  | --- CAR_VS_BOS_001 
|  | --- CAR_VS_NYR__001
|  | ---
|  | ---
|     
| --- train
|  | --- CAR_VS_BOS_002
|  | --- CAR_VS_NYR_002
|  | --- 
|  | ---
|
| --- validation 
|  | --- PIT_VS_WAS_001
|
./utilities.py
```


### Acknowledgement:
We're thankful to the work done by Zhao et al. [MHPTD](https://github.com/grant81/hockeyTrackingDataset) for open-sourcing the first version of this dataset. Our work is built after relevant modifications and error corrections, inheriting the
same licensing (GPL-3.0) and following any and all restrictions described in it. Our only hope with this work is to encourage ice hockey MOT research and make it publicly accessible. 

### Citation: 
If you use this work, please cite: 

```text
@inproceedings{prakash2024vip,
  title={VIP-HTD: A Public Benchmark for Multi-Player Tracking in Ice Hockey},
  author={Prakash, Harish and Chen, Yuhao and Rambhatla, Sirisha and Clausi, David and Zelek, John},
  booktitle={Computer Vision and Intelligent Systems},
  year={2024}
}
```
