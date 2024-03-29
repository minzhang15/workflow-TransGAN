# Workflow detection with improved phase discriminability
Min Zhang, Haiyang Hu, Zhongjin Li
# Introduction
The temporal-aware workflow detection framework is a unified Transformer and graph attention network framework for workflow detection in videos, which uses I3D to extract features.

mAP results on POTFD

_tIoU_   |  0.4 | 0.5 | 0.6  | 
:-------|:-----:|:-----:|:-----:|
| R-C3D [42] | 82.6 | 79.5 | 73.2 |
TAL-Net [43] | 88.4 |85.7 | 79.8 | 
GTAN [44]    | 87.8 | 84.2 | 78.3 |
A2Net [45]   | 89.5 |85.9 | 80.1 |
PCG-TAL [46] | 90.4 | 86.7 | 81.3 |
AFSD [36]    | 92.4 | 87.2 | 82.7 |
**TransGAN(ours)** | **93.7** | **89.5** | **83.1** |

Comparison to State-of-the-art methods on THUMOS-14

_tIoU_   |  0.1 | 0.2 | 0.3  | 0.4 | 0.5 |
 :-------|:-----:|:-----:|:-----:|:-----:|:-----:|
R-C3D [42]   | 54.5 | 51.5 | 44.8 |35.6 | 28.9 |
TAL-Net [43] | 59.8 | 57.1 | 53.2 | 48.5 | 42.8 |
GTAN [44]    | 69.1 | 63.7 | 57.8 |47.2 | 38.8 |
A2Net [45]   | 61.1 | 60.2 | 58.6 |54.1 | 45.5 |
PCG-TAL [46] | 71.2 | 68.9 | 65.1 |59.5 | 51.2 |
AFSD [36]    | -    | -    | 67.3 |62.4 | 55.5 |
**TransGAN(ours)** | **70.4** | **69.3** | **68.0** |63.5 | 57.6 |

# Contents
1. Installation
2. Datasets
3. Training
4. Testing
5. Evaluation

# Installation
1. Get the code. We will call the directory that you cloned Caffe into $CAFFE_ROOT.<br />
      git clone https://github.com/minzhang15/workflow-detection.git<br />
      cd caffe<br />
      git checkout workflow-detector

2. Build the code. Please follow Caffe instruction to install all necessary packages and build it.<br />
            # Modify Makefile.config according to your Caffe installation.<br />
            cp Makefile.config.example Makefile.config<br />
            make -j8<br />
            # Make sure to include $CAFFE_ROOT/python to your PYTHONPATH.<br />
            make py<br />
            make test -j8<br />
            # (Optional)<br />
            make runtest -j8
            
# Datasets
      To download the ground truth tubes, run the script: 
      ./cache/fetch_cached_data.sh ${dataset_name} # dataset_name: POTFD, THUMOS, UCF101
This will populate the cache folder with three pkl files, one for each dataset. For more details about the format of the pkl files, see enhanced-detector-scripts/Dataset.py.     

If you want to reproduce exactly our results as reported in Tables 3 and 4, we also provide the RGB and flow files for the three datasets we use.

1. POTFD
You can download the frames (4.2GB) and ground truth annotations:

            ./data/POTFD/get_jhmdb_data.sh number # number = 0 for for RGB Frames 

3. THUMOS-14
You can download the frames (4.4GB) and ground truth annotations:

            ./data/THUMOS/get_ucf101_data.sh number # number = 0 for for RGB Frames

These will create the Frames and FlowBrox04 folders in the directory of each dataset.

# Training
1. We provide the prototxt used for our experiments for POTFD and THUMOS-14. These are stored in:<br />
caffe/models/workflow-detection/${dataset_name}.

2. Download the RGB initialization models pre-trained on Kinetics dataset:

            ./models/workflow-detection/scripts/fetch_initial_models.sh
            
This will download the caffemodels: caffe/models/workflow-detection/initialization_i3d_RGB.caffemodels and caffe/models/workflow-detection/initialization_i3d_FLOW5.caffemodels

3. We provide an example of training commands for a ${dataset_name}:
(a) RGB

            export PYTHONPATH="$./workflow-detection-scripts:$PYTHONPATH"                       # path of detector 
            ./build/tools/caffe train \
            -solver models/workflow-detection/${dataset_name}/solver_RGB.prototxt \             # change dataset_name 
            -weights models/workflow-detection/initialization_i3d_RGB.caffemodel \
            -gpu 0                                                                        # gpu id

(b) stacked Flows

            export PYTHONPATH="$./worflow-detection-scripts:$PYTHONPATH"                       # path of workflow-detection 
            ./build/tools/caffe train \
            -solver models/workflow-detection/${dataset_name}/solver_FLOW5.prototxt \           # change dataset_name 
            -weights models/workflow-detection/initialization_i3d_FLOW5.caffemodel \
            -gpu 0                                                                        # gpu id

where ${dataset_name} can be: POTF, THUMOS-14.

# Testing
1. If you want to reproduce our results for the POTF, THUMOS-14 datasets, you need to download our trained caffemodels. To obtain them for sequence length K=6, run from the main caffe directory for each dataset:

            ./models/workflow-detection/scripts/fetch_models.sh ${dataset_name} # change dataset_name 
This will download one RGB.caffemodel and one FLOW5.caffemodel for each dataset. These are stored in models/workflow-detection/${dataset_name}.

2. Next step is to extract tubelets. To do so, run:

            python workflow-detection-scripts/ENDtor.py "extract_tubelets('${dataset_name}', gpu=-1)" # change dataset_name, -1 is for cpu, otherwise 0,...,n for your gpu id 
The tubelets are stored in the folder called workflow-detection-results. Note that the test is not efficient and can be coded more efficiently by extracting features once per frame.

3. For creating tubes, you can run the following:

            python workflow-detection-scripts/ENDtor.py "BuildTubes('${dataset_name}')"     # change dataset_name 
The tubelets are stored in the folder called results/workflow-detection.

For all cases ${dataset_name} can be: POTFD, THUMOS14.

# Evaluation
1. For evaluating the per-frame detections, we provide scripts for frame-mAP and frame-Classification. You can run them as follows:

            python workflow-detection-scripts/ENDtor.py "frameAP('${dataset_name}')"       # change dataset_name
            python workflow-detection-scripts/ENDtor.py "frameCLASSIF('${dataset_name}')"

2. For evaluating the tubes, we provide scripts for video-mAP. You can run it as follows:

            python workflow-detection-scripts/ENDtor.py "videoAP('${dataset_name}')"       # change dataset_name
