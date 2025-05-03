# DeepFake Detection
## Step 1: Xception
- Image Model
- Platform: Kaggle
- V1: Created the detection algorithm using Xception as the model and see the results;
- V2: Add Grad-CAM as Explainable AI;
- Conclusion: the model should do better previsions.
- Next steps: Improvents to achieve better results
## Step 2: [Liveness Classifier](https://www.kaggle.com/models/kameshrasu/liveness_classifier/)
- Video Model from Kaggle
- Platform: Kaggle
- V1: Created a simple detection script, it should do better previsions
- V2: Apply face extrations using different methods. Improvement in previsions and best results using YOLO, MTCNN and SSD
- V3: Add Xplainable AI, Frequency and Occlusion Sensitivity Analysis
- Next steps: Improve model and face ebbeding
## Step 3: Experiments using GradCAM
Part 01
- Replicate Hybrid (CNN + PCA + SVM) and GAN models
- Evaluate model
- Apply GradCAM in dataset with 6 deepfake generation types

Part 02
- Fix script applying frame struct, not videos
## Step 4: Experiments using Innvestigate
Part 01
- Kaggle Notebook that implements Innvestigate: [LRP_method](https://www.kaggle.com/code/achintyabhat/lrp-method)
- Add innvestigate to VGG16 model
- TODO: fix innvestigate in resnet and efficientnet models
## Script
1. Extract face from Faceforensics dataset
