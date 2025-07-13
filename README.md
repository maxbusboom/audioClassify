# Audio Classification Project

This project implements audio classification using CNN models on the Urban Sound 8K dataset.

## Setup

### 1. Environment Configuration

The project uses environment variables for configuration. A `.env` file is already configured with:

```bash
DATA_DIR=/Users/max/proj/audioClassify/data
```

### 2. Dataset Download

To download the Urban Sound 8K dataset, use the provided script:

```bash
cd data
./download_urban_sounds.sh
```

This script will:
- Download the Urban8K dataset from the official source
- Extract the files to the data directory
- Set up the proper directory structure

### 3. Directory Structure

After downloading, your project should look like:

```
audioClassify/
├── data/
│   ├── UrbanSound8K/
│   │   ├── audio/
│   │   │   ├── fold1/
│   │   │   ├── fold2/
│   │   │   └── ... (fold10)
│   │   └── metadata/
│   │       └── UrbanSound8K.csv
│   └── download_urban_sounds.sh
├── notebooks/
│   └── maxbusboom/
│       └── urban_8k_cnn_models.ipynb
├── .env
└── README.md
```

## Usage

1. **Load environment variables**: The notebook automatically loads the `.env` file using `python-dotenv`
2. **Check data availability**: The notebook will verify that the dataset is available
3. **Run analysis**: Execute the cells to perform audio classification

## Dataset Information

The Urban Sound 8K dataset contains 8,732 labeled sound excerpts (≤4s) from 10 classes:
- air_conditioner
- car_horn
- children_playing
- dog_bark
- drilling
- engine_idling
- gun_shot
- jackhammer
- siren
- street_music

## Features

- **MFCC Feature Extraction**: Extract Mel-frequency cepstral coefficients from audio files
- **CNN Models**: Multiple CNN architectures (4, 6, and 8 layer models)
- **Dynamic Model Architecture**: Customizable CNN with configurable layers
- **Performance Visualization**: Training/validation loss and accuracy plots

## Requirements

The project requires Python packages listed in `requirements.txt`. Key dependencies include:
- librosa (audio processing)
- torch (deep learning)
- pandas (data manipulation)
- matplotlib/seaborn (visualization)
- python-dotenv (environment variables)