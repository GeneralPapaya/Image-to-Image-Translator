# Pix2Pix Image-to-Image Translation

This project uses a Pix2Pix (Generative Adversarial Network) model to perform image-to-image translation. The model is trained to learn the mapping from a "raw" input image to a corresponding "edited" target image.

---

## Model

The pre-trained generator model is saved as `GAN_Raw_image_2_edit.h5`. You can download it from the link below:

* **Model Download:** [https://drive.google.com/file/d/1xoeAPbiSiExlIIpNDZ32tVdTAjXGIWiG/view?usp=sharing](https://drive.google.com/file/d/1xoeAPbiSiExlIIpNDZ32tVdTAjXGIWiG/view?usp=sharing)

---

## Dataset

The model was trained on the **Adobe-FiveK** dataset. This dataset contains 5,000 raw images and their corresponding retouched (edited) versions by experts.

* **Dataset Source:** [https://www.kaggle.com/datasets/weipengzhang/adobe-fivek](https://www.kaggle.com/datasets/weipengzhang/adobe-fivek)

---

## Project Notebooks

This repository contains two main Jupyter notebooks:

1.  **`Pix2Pix Model.ipynb`**: This notebook contains the complete pipeline for building, training, and saving the Pix2Pix generator and discriminator models using the Adobe-FiveK dataset.
2.  **`Main.ipynb`**: This notebook is used for inference. It loads the pre-trained generator model and runs it on sample images to visualize the *Raw Image* (input), *Edit Image* (ground truth), and *Predicted Image* (model output).

---

## How to Use (Inference) 🚀

To run the model and see predictions:

1.  **Download the Model:** Use the link above to download the `GAN_Raw_image_2_edit.h5` file.
2.  **Place the Model:** Place the downloaded `.h5` file in your project directory.
3.  **Open `Main.ipynb`**: Open the `Main.ipynb` notebook in a Jupyter environment.
4.  **Update Path:** Make sure the path in the `load_model()` function correctly points to the location of the `.h5` file.
5.  **Run Predictions:** Execute the notebook cells to see the model generate predictions.

---

## Dependencies

The project primarily uses the following Python libraries:
* TensorFlow / Keras
* NumPy
* Matplotlib
