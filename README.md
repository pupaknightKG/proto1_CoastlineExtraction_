# Automated Coastline Extraction using Sentinel-1 SAR & U-Net (Prototype 1)

This project provides an automated deep learning pipeline to detect and map coastlines from European Space Agency (ESA) Sentinel-1 Synthetic Aperture Radar (SAR) satellite imagery downloaded from CDSE Browser. The workflow begins with `01_preprocess_santa_barbara.ipynb` to calibrate raw radar data, filter speckle noise, and generate sliced image tiles with Otsu ground-truth masks, followed by `notebookf600e8c2ce.ipynb` to train a PyTorch U-Net neural network that produces the trained weights (`unet_coastline_weights_sb.pth`).

For complete technical documentation, formulas, and configuration details, see `README_DETAILED.md`.

**Status & Sequence Note:** The pipeline is designed to be executed sequentially starting with preprocessing (`01_preprocess_santa_barbara.ipynb`) and continuing with model training (`notebookf600e8c2ce.ipynb`). Please note that the final stage—the model inference and coastline vectorization notebook (`01_inference_santa_barbara.ipynb`)—is currently undergoing active fixing and refinement. Till then, ignore README_DETAILED.
