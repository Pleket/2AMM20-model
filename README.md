# 2AMM20-model
This is the code for Group 3 from the course "Research Topic in Data Mining", 2023/2024. The paper is titled "don't forget to add title".

## Dataset
Download the dataset from [this link](https://downloads.cs.stanford.edu/nlp/data/dro/waterbird_complete95_forest2water2.tar.gz). This file contains 200 folders for the 200 species of birds in the dataset, and a file metadata.csv specifying whether each image depicts a water or a landbird, and whether it has been given a water or a land background. After extracting the file, function `organize_bird_images` will organize the data into 2 folders, one for landbirds and one water waterbirds, which will be used for the rest of the code.

## Running the Method
Everything required to reproduce our experiments can be found in the file `JTT_CVaR.ipnyb`. Hyperparameters such as the number of epochs, learning rate and batch size can be varied at the bottom of the code. Parameter CVaR can also be adjusted to either True of False depending on whether CVaR loss should be used in the training process. Visualisation of the results is also implemented.
