# Ego2Hands: 

<img src="imgs/00198_gt_vis_seq2.png" width="320">    <img src="imgs/00198_gt_vis_seq4.png" width="320">

<img src="imgs/00198_gt_vis_seq5.png" width="320">    <img src="imgs/00198_gt_vis_seq7.png" width="320">

Ego2Hands is a large-scale dataset for the task of two-hand segmentation/detection in unconstrained environments. The training set provides images of only the right hands from 22 subjects with segmentation and hand energy ground truth annotation, allowing compositing-based data generation for unlimited training data with segmentation and detection ground truth. The evaluation set provides 8 sequences from 4 subjects, covering different scenes, skin tones and various level of illumination. This dataset is introduced by our paper [Ego2Hands: A Dataset for Egocentric Two-hand Segmentation and Detection](https://arxiv.org/abs/2011.07252). 

See our [Youtube Demo](https://www.youtube.com/watch?v=WjmPgnDXiMA&ab_channel=AlexLin) for a sample application in color-based gesture control.
<img src="imgs/demo.webp" width="480">

## Convolutional Segmentation Machine
We introduce a well-balanced architecture with compact model size, fast inference speed and high accuracy for real-time two-hand segmentation/detection. The script for training and testing is provided in this repository. To run the script, please follow the instructions below:
