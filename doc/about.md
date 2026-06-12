# Repository Structure
In this section you can find the repository structure and a short description of the relevant files. 

The project is structured to provide a clear separation between data processing, model implementation, training, evaluation, and deployment. The main source code is located in the `src/` directory, while trained model weights, datasets, tests, and documentation are organized into dedicated folders for ease of use and development.


``` text
├── datasets/
├── src/
  ├── data.py            # Dataset loaders (per-customer and general)
  ├── evaluation.py      # Evaluation script (precision, recall, speed)
  ├── inference.py       # Entry point for running predictions
  ├── instructions.md    # Instructions to use the scripts and extend the code
  ├── models.py          # Model definitions (per-customer and general)
  └── training.py        # General training script
├── tests/
  └── test_training.py   # Testing of the basic functionalities of src/training.py
├── weights/             # Trained model weights (trained on LUMI)
├── README.md            #
└── environment.yml      # Environment configuration file      
```
