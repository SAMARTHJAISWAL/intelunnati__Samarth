# intelunnati__Samarth

# Clothing Image Classification using CNN

This repository contains the code and documentation for a project on clothing image classification using Convolutional Neural Networks (CNNs). The project aims to develop a model that accurately classifies clothing images into their respective categories.

## Directory Structure

The repository is organized into the following folders:

models: Contains trained model files.
data: Contains the dataset used for training and testing. The dataset is already included in this folder.
code: Contains the code files related to the project. It includes scripts, notebooks, and other code components.
docs: Contains documentation files related to the project.
report.md: Conclusion of tasks performed, mentioning tests carried out, and the final accuracy achieved.
approach.md: Contains the steps taken to reach the final outcome, along with the outcomes and reasoning for each approach taken.
demo_videos: Contains video demonstrations of the final working project. If there are multiple approaches, there will be separate videos for each.

## How to Run the Code

To run the code, please follow the steps below:

### Set up the Environment

Ensure that Python 3.7 or higher is installed on your system. Install the necessary libraries and dependencies by running the following command:

pip install -r requirements.txt

### Model Training

1. Open the code file `clothing_classification.py` in your preferred Python IDE or editor.
2. Run the code to train the model. The trained model will be saved in the `models` folder.

### Model Evaluation

1. After training, the code will automatically evaluate the model on the test set and display the test accuracy.
2. Classification reports, confusion matrices, and ROC curves will be generated and saved in the `docs` folder.

### Visualize Model Architecture

To visualize the model architecture, make sure you have the `visualkeras` package installed. If not, install it using:

```
pip install visualkeras
```

Then, run the following code:

```python
import visualkeras
from tensorflow.keras.models import load_model

model = load_model('models/your_model_file.h5')  # Replace 'your_model_file.h5' with the actual model file name

# Visualize the model
visualkeras.layered_view(model)
```

### Optional: IoU Calculation

If you have binary segmentation tasks, you can calculate the Intersection over Union (IoU) by running the provided `calculate_iou` function on the test predictions. The average IoU across all test images will be displayed.

### Demo Videos

Check the demo_videos folder to find video demonstrations of the final working project. If there are multiple approaches, there will be separate videos for each.

Feel free to explore the code, documentation, and model files to understand the project in detail.

For any questions or issues, please reach out to the repository owner.
Check the `demo_videos` folder to find video demonstrations of the final working project. If there are multiple approaches, there will be separate videos for each.

Feel free to explore the code, documentation, and model files to understand the project in detail.

For any questions or issues, please reach out to the repository owner.

Happy coding!
