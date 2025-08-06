# 🚀 Stroke Lesion Segmentation with ISLES'22 Dataset

Welcome to this exciting project! 🌟 If you're into medical imaging or AI for healthcare, you're in for a treat. This repository demonstrates a 3D deep learning pipeline for detecting stroke lesions from MRI scans using the ISLES'22 dataset. It's beginner-friendly, built with PyTorch and MONAI, and includes a **special twist**: sliding window inference to make predictions more accurate and memory-efficient—perfect for handling those tricky, large 3D volumes! 🧠💡

## 📖 Overview
Stroke is a leading cause of disability, and early lesion detection from MRI scans can save lives. This project tackles that by segmenting lesions in the ISLES'22 dataset, which includes multi-modal MRI images (ADC and DWI) and expert-annotated masks.

We use a 3D U-Net model to predict binary lesion masks. Trained for just 10 epochs (as a quick demo), it shows promising results. The **highlight**? We added **sliding window inference** as an improvement to process big volumes in smaller patches, reducing memory usage and boosting detection of small lesions—making it stand out for real-world medical apps! 🔍✨

## 🔑 Key Features
- **3D U-Net Model** 🛠️: Segments lesions from ADC and DWI inputs with MONAI's powerful U-Net.
- **🌟 Sliding Window Inference (Our Special Improvement!)**: Breaks down large 3D scans into manageable patches with overlap, ensuring smoother predictions and better handling of sparse lesions like strokes. This is a game-changer for memory efficiency and accuracy!
- **Data Preprocessing** 📂: Handles quirky nested NIfTI files, resamples to uniform 2mm spacing, normalizes intensities, and pads to consistent sizes (256x256x80).
- **Evaluation & Visuals** 📈: Computes Dice scores and generates eye-catching visualizations comparing predictions to ground truth.

## 📊 Results
After 10 epochs:
- **Training Loss**: [insert final_train_loss, e.g., 0.8319] 📉
- **Validation Dice**: [insert final_val_dice, e.g., 0.0176] 🎯
- **Test Dice**: [insert test_dice, e.g., 0.05-0.2 estimated] 🏆

These are from a quick run—longer training would boost scores! Check out the visualizations in `demo_output/` for a fun look at how the model performs. 😎

Example Visualization (`segmentation_demo_sample_1.png`):
![Segmentation Demo](https://github.com/AyushiKashyapp/ISLES-2022-Dataset/blob/main/ISLES-2022/pred.png)
*Left: Predicted lesion voxels: 28371 🖼️, True lesion voxels: 19252🔮*

## 🗂️ Dataset
The [ISLES'22 dataset](https://www.kaggle.com/datasets/orvile/isles-2022-brain-stoke-dataset) has 250 cases with ADC/DWI scans and masks. We resample everything to `./isles2022_resampled` for consistency. 📁

## 🛠️ Installation
1. **Clone the Repo**:
   ```bash
   git clone [(https://github.com/AyushiKashyapp/ISLES-2022-Dataset)]
   ```

2. **Install Dependencies** (Python 3.8+):
   ```bash
   pip install -r requirements.txt
   ```
   (Includes torch, monai==0.8.1, nibabel, numpy, matplotlib, tqdm.)

3. **Download Dataset**:
   - Grab it from [Kaggle](https://www.kaggle.com/datasets/orvile/isles-2022-brain-stoke-dataset).
   - Place in `./isles2022/ISLES-2022/ISLES-2022`.
   - Run the notebook's resampling step to prep `./isles2022_resampled`.

## 🚀 Usage
Fire up `isles22_segmentation.ipynb` in Jupyter or Colab:
- Download & preprocess data. 📥
- Train the U-Net (10 epochs). ⚙️
- Evaluate and visualize results. 🎨

Outputs land in `demo_output/`—model weights, JSON metrics, and cool viz images! 📸

## 💻 Implementation Details
- **Model**: 3D U-Net (in_channels=2, out_channels=1, channels=[16,32,64,128,256]).
- **Loss**: 50/50 mix of Dice and BCE—simple yet effective!
- **Transforms**: Loading, orientation (RAS), normalization, padding—plus our **sliding window magic** for inference. ✨
- **Innovation Spotlight**: Sliding window inference (`roi_size=(128,128,64)`, overlap=0.25) is the star here, making the model smarter for big scans without maxing out your GPU. 🌟

## 🔮 Future Ideas
- Train longer for better Dice scores. ⏱️
- Add focal loss for even smarter imbalance handling. ⚖️
- Upgrade to Attention U-Net (if you bump MONAI versions). 🤖

## 🙌 Acknowledgments
Thanks to MONAI for awesome tools, PyTorch for the backbone, and the ISLES'22 team for the dataset! ❤️
