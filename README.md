# Ego2Hands

<img src="imgs/00198_gt_vis_seq2.png" width="320">    <img src="imgs/00198_gt_vis_seq4.png" width="320">

<img src="imgs/00198_gt_vis_seq5.png" width="320">    <img src="imgs/00198_gt_vis_seq7.png" width="320">

Ego2Hands is a large-scale dataset for the task of two-hand segmentation/detection in unconstrained environments. The training set provides images of only the right hands from 22 subjects with segmentation and hand energy ground truth annotation, allowing compositing-based data generation for unlimited training data with segmentation and detection ground truth. The evaluation set provides 8 sequences from 4 subjects, covering different scenes, skin tones and various level of illumination. This dataset is introduced by our paper [Ego2Hands: A Dataset for Egocentric Two-hand Segmentation and Detection](https://arxiv.org/abs/2011.07252). 

See our [Youtube Demo](https://www.youtube.com/watch?v=WjmPgnDXiMA&ab_channel=AlexLin) for a sample application in color-based gesture control.
<img src="imgs/demo.webp" width="480">

## Convolutional Segmentation Machine
We introduce a well-balanced architecture with compact model size, fast inference speed and high accuracy for real-time two-hand segmentation/detection. The implementation for training and testing is provided using Pytorch. 

To run the script, please follow the instructions below:

1. **Download the repository**
2. **Download the Ego2Hands dataset**
  * This dataset is about 90GB in size as its training set contains ~180k images with segmentation and energy, and its evaluation set contains 2k fully annotated images. Make sure you have enough space for this project. 
  * Use the following download links and put the data in the proper path:
  
    * **Ego2Hands (train):** [subject0-4](https://byu.box.com/s/moy2j92p9j9tv8mw8c1dgafn4r4pod19), [subject5-10](https://byu.box.com/s/jdto18tt4q89pdmn2l2wiiics2ltdr54), [subject11-16](https://byu.box.com/s/0yj1iqlsmt7aw7odp3ns50e39nmer4vo), [subject17-21](https://byu.box.com/s/fr3lcjscu5xit6qbyqdooy6pi6uyk1q3)

    Move the training data into directory "/data/Ego2Hands/train/". The "/train/" folder should contain the subjects' folders. 
    
    * **Backgrounds:** [Download link](https://byu.box.com/s/dc16feb1nhswm3imtce7f6r5ai7d0i6w)
    
    Move the background data into directory "/data/Ego2Hands/backgrounds". The background images are collected from online sources with free license and we do not own rights for the background images. We also used the images from the DAVIS 2016 and 2017 dataset as background images. Please download them through https://davischallenge.org/ and extract the images into the "/data/Ego2Hands/backgrounds" directory as well. DAVIS datasets should have 210 sequences and 14997 images available as backgrounds (we do not use their segmentation annotation). If you use the DAVIS data, please abide by their term of use. Also feel free to include your own background images as well since they are rather easy to collect (the script recursively finds all files that ends with '.jpg' and '.png' in the background directory as background images).
    
    * **Ego2Hands (eval):** [subject22-25](https://byu.box.com/s/ys2a83r8iga0tlh7aogesc1g1i49jsur)

    Move the evaluation data into directory "/data/Ego2Hands/eval/". The "eval/" folder should contain the sequence folders and their corresponding background folders.
 
3. **Download the pretrained models (Optional)**
We provide [pretrained models](https://byu.box.com/s/t30xmoum43c4fdctjvdk72wba62n6b6o) with input edge channel and energy output channel as well as the scene-adapted models for the 8 evaluation sequences. You can test the models' performance by copying the provided models into the correct directory and following the instructions for testing below.
 
4. **Environment Setup**
We used the following steps to set up the proper environment in Anaconda on a Windows machine:
  > conda create --name ego2hands_env python=3.7\
  > conda activate ego2hands_env\
  > conda install -c anaconda numpy\
  > conda install -c conda-forge opencv\
  > conda install pytorch torchvision torchaudio cudatoolkit=10.1 -c pytorch (see https://pytorch.org/ for a proper setting specific for your machine)\
  > conda install -c conda-forge easydict\
  > conda install -c intel pyyaml

5. **Usage:**\
Run the following code for testing different functionalities using the arguments below:

  * **Training**
 
    - [x] Input edge channel
    - [x] Output energy channel
    - [ ] Training multiple models
    - [ ] Domain adaptation
    - [x] Saving outputs for visualization.
 
    > **python model_train_test.py --config configs\config_ego2hands_csm.yml --input_edge --energy --save_outputs**
    
    * Modify the arguments below in the config file for actual training. We used the following values for our experiments.
    
      > max_iter_seg: 100000\
      > max_iter_seg_adapt: 10000\
      > display_interval: 1000\
      > save_interval: 2000
    * Trained models will be saved in "models_saved" folder. Outputs will be saved in the "outputs" folder.
    * You can also train models without the input edge channel or energy output channel (note that **"--energy" will also set "--input_edge" because the energy feature is applied incrementally**).
    * Every "display_interval" iterations, the script displays training information and outputs visualization for the current batch in the "outputs" folder if "--save_outputs" is set.
    * Every "save_interval" iterations, the script evaluates the model on the evaluation sequences. Testing results will also be saved in "outputs" folder if "--save_outputs" is set.
    
  * **Testing**
  
    - [x] Input edge channel
    - [x] Output energy channel
    - [ ] Testing multiple models
    - [ ] Domain adaptation
    - [x] Saving outputs for visualization.
  
    > **python model_train_test.py --config configs\config_ego2hands_csm.yml --input_edge --energy --eval --save_outputs**
  
    * Evaluation results for all 8 evaluation sequences will be displayed in the terminal for model with the corresponding setting (using input edge map and energy output channel).
    * Output visualization for all test images will be saved in the "outputs" directory.
    
  * **Training and testing multiple models**
  
    - [x] Input edge channel
    - [x] Output energy channel
    - [x] Training multiple models
    - [ ] Domain adaptation
    - [x] Saving outputs for visualization.
  
    > **python model_train_test.py --config configs\config_ego2hands_csm.yml --input_edge --energy --train_all --num_models 3 --save_outputs**
    
    * "--train_all" and "--num_models" need to be used to train the specified amount of model instances. As training images are not fixed and are composited in training time, each model experiences different training instances and can have variance. In our experiments, most models have stable performance after training for 100k iterations with batch size of 4. 
    
    - [x] Input edge channel
    - [x] Output energy channel
    - [x] Testing multiple models
    - [ ] Domain adaptation
    - [x] Saving outputs for visualization.
    
    > **python model_train_test.py --config configs\config_ego2hands_csm.yml --input_edge --energy --test_all --num_models 3 --save_outputs**
    
    * "--test_all" automatically sets "--eval" and will evaluate the number of models specified by "--num_models". If the script cannot find the saved model that ends with "pretrained.pth.tar", it skips the model because this indicates that this pretrained model for testing does not exist.
    
  * **Domain adaptation**
  
    - [x] Input edge channel
    - [x] Output energy channel
    - [x] Training multiple models
    - [x] Domain adaptation
    - [x] Saving outputs for visualization.
    
    > **python model_train_test.py --config configs\config_ego2hands_csm.yml --input_edge --energy --train_all --num_models 3 --adapt --save_outputs**
    
    * "--adapt" sets the script in adaptation mode. For training, adaptation requires models to be pretrained first.
    * Adaptation mode replaces the random training backgrounds with background images for that specific sequence during data generation.
    * Adapted model for each sequence will be saved in the corresponding save location for the specified setting. 
    * Output visualization will be saved in the "outputs" folder if "--save_outputs" is set.
    * If all 3 pretrained models are available, this will train 3 * 8 = 24 scene-adapted models. 
    
    - [x] Input edge channel
    - [x] Output energy channel
    - [x] Testing multiple models
    - [x] Domain adaptation
    - [x] Saving outputs for visualization.
    
    > **python model_train_test.py --config configs\config_ego2hands_csm.yml --input_edge --energy --test_all --num_models 3 --adapt --save_outputs**
    
    * This will test all scene-adapted models and display the evaluation results.
    * Output visualization will be saved in the "outputs" folder if "--save_outputs" is set. 
    * Adapted models should outperform the pretrained models.
    
  * **Speed test**
  
    * Setting the "--speed_test" will evaluate the model on the evaluation sequences and skip output visualization as well as score evaluation. Instead it times the time it takes for the model to make an estimation and display the averaged results.
    
  * **Custom testing**
  
    To adapt a pretrained model on your custom sequence, please do the following:
    
    * Record a background sequence (no hands) for your custom scene. Move the data into "/data/Ego2Hands/custom/custom_train/". When recording, please try to collect data with variety that properly represent the actual testing environment (for example, move back & forth, left & right, look up & down, left & right slightly. You could add in some slight rotation as well.) For the demo video, we collected the background sequence with the screen playing a video of the gameplay.
    * Record a testing sequence (with hands) for your custom scene. Move the data into "/data/Ego2Hands/custom/custom_eval/".
    * Check the path in the config file to see if "custom_bg_dir" and "custom_eval_dir" correctly lead to the saved data.
    * Make sure "models_saved/ego2hands/CSM/with_energy/1/ego2hands_CSM_seg_pretrained.pth.tar" exists for adaptation. If you desire to run custom adaptation on settings other than "with_energy" (we provide pretrained models with input_edge and energy), you would need to pretrain the model yourself with the desired setting first. In our experiments, CSM with input edge and energy has the best performance.
    
    To adapt to the custom scene, run:
    
      > **python model_train_test.py --config configs\config_ego2hands_csm.yml --input_edge --energy --custom --save_outputs**
    
      * This will use the first instance of the trained model with input edge and energy as pretrained model and adapt on the custom scene. "--custom" setting does not support multiple models as it is only for quick evaluation. The custom-adapted model will be saved as "models_saved/ego2hands/CSM/with_energy/1/ego2hands_CSM_seg_custom_pretrained.pth.tar".
    
    To test on the custom scene, run:
    
      > **python model_train_test.py --config configs\config_ego2hands_csm.yml --input_edge --energy --eval --custom**
    
      * This will use the custom-adapted model to evaluate on the collected test sequence. Output visualization will be saved in "outputs/ego2hands_CSM_edge1_energy1_seg_test_custom" regardless of the setting of "--save_outputs".
      
    To test on the custom scene using only the pretrained model without custom domain adaptation, just copy the pretrained model at"models_saved/ego2hands/CSM/with_energy/1/ego2hands_CSM_seg_custom_pretrained.pth.tar" and rename the copy as "ego2hands_CSM_seg_custom_pretrained.pth.tar". The pretrained model is capable of achieving certain accuracy but scene-adapted model definitely has better performance.
    
## Gesture Control
We collected a small [gesture dataset](https://byu.box.com/s/3m0u0jwepac6ot01p0xtrqknx6lbvbna) (2 subjects, 5 static gestures). The "gesture_annotations.txt" file consists of all the bounding box, label, image path info needed to extract the squared bounding boxes for creating input images.
    
We trained a very simple classifier (Resnet with only 2 downsampling layers each with 1 block followed by fully connected layer) to classify 5 classes given a cropped hand segmentation image (64x64, binary input). Feel free to train your classifier using our gesture dataset for real-time gesture control. 

For overwriting the keyboard keys with python control, please keep in mind that games that use DirectX need "PyDirectInput" for key controls and mouse clicks. To simulate mouse movement (with speed instead of instant location change), try this [solution](https://stackoverflow.com/questions/56386470/passing-mouse-coordinates-from-pyautogui-to-direct-input-causes-the-the-mouse-to). **Remember that both the operating command prompt and the game need to be opened with administrator priviledges to enable python control.**

## License
This dataset can only be used for scientific/non-commercial purposes. If you use this dataset in your research, please cite the corresponding [paper](https://arxiv.org/abs/2011.07252).
