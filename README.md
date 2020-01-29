# Adversarial Image-to-Frequency Transform Network
Toy example for 'Adversarially Learnt Image to Frequency Transform Network (AIFT)' on the paper (
Unsupervised Pixel-level Road Defect Detection via Adversarial Image-to-Frequency Transform) (Implemented with Py36 and Pytorch)
![](readme/test.png)


* The paper is submitted to IV2020, and it is under the review.
* This source code is a toy example for AITF, and it does not include the evaluation code for the experiments on the paper.
Contact: [jm.andrew.yu@gmail.com] Any questions or discussions are welcomed! 


## Abstract.
In the past few years, the performance of road defect detection has been remarkably improved thanks to advancements on various studies on computer vision and deep learning. Although a large-scale and well-annotated datasets enhance the performance of detecting road pavement defects to some extent, it is still challengeable to derive a model which can perform reliably for various road conditions in practice, because it is intractable to construct a dataset considering diverse road conditions and defect patterns. To end this, we propose an unsupervised approach to detecting road defects, using Adversarial Image-to-Frequency Transform (AIFT). AIFT adopts the unsupervised manner and adversarial learning in deriving the defect detection model, so AIFT does not need annotations for road pavement defects. We evaluate the efficiency of AIFT using GAPs384 dataset, Cracktree200 dataset, CRACK500 dataset, and CFD dataset. The experimental results demonstrate that the proposed approach detects various road detects, and it outperforms existing state-of-the-art approaches.

## File configuration
.<br>
├── frequency_discriminator.pkl<br>
├── image_discriminator.pkl<br>
├── inception_score_graph.txt<br>
├── logs<br>
│   └── events.out.tfevents.1578130324.neumann-System-Product-Name<br>
├── main.py<br>
├── model<br>
│   ├── aiftn.py<br>
│   └── __pycache__<br>
│       └── aiftn.cpython-36.pyc<br>
├── negative_generator.pkl<br>
├── positive_generator.pkl<br>
├── README.md<br>
├── src<br>
│   ├── config.py<br>
│   ├── dataset.py<br>
│   ├── __init__.py<br>
│   ├── __pycache__<br>
│   │   ├── config.cpython-36.pyc<br>
│   │   ├── dataset.cpython-36.pyc<br>
│   │   ├── __init__.cpython-36.pyc<br>
│   │   ├── tensorboard_logger.cpython-36.pyc<br>
│   │   └── utils.cpython-36.pyc<br>
│   ├── tensorboard_logger.py<br>
│   └── utils.py<br>
├── tb_log.txt<br>
├── test.png<br>
├── _t_main.py<br>
└── training_result_vis<br>


## How to train
~~~
python main.py
~~~


## Curves of the cost functions on AITF. 
(Loss for Discriminator, generator for positive phase, generator for negative phase)

<p align="center">  <img src='readme/loss_d.svg' align="center" height="180px"> <img src='readme/loss_gn.svg' align="center" height="180px"> <img src='readme/loss_gp.svg' align="center" height="180px">  </p>


## Given and transformed samples (40000 Iterations)
Given image and frequency samples.
<p align="center">  <img src='results/img_iter_40000.png' align="center" height="300px"> <img src='results/freqs_iter_40000.png' align="center" height="300px">  </p>


Transformed image and frequency samples.
<p align="center">  <img src='results/img_transformed_iter_40000.png' align="center" height="300px"> <img src='results/freq_transformed_iter_40000.png' align="center" height="300px">  </p>


bibtex as follows:

