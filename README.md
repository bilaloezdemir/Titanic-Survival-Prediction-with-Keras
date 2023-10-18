# Titanic Survival Prediction with Deep Learning

This project uses deep learning to predict the survival of passengers on the Titanic. It involves data preprocessing, model building, and training. The code is written in Python and utilizes libraries like pandas, scikit-learn, Keras, and matplotlib.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Data Preprocessing](#data-preprocessing)
- [Model Building](#model-building)
- [Training and Visualization](#training-and-visualization)
- [Contributing](#contributing)
- [License](#license)


## Usage

1. Download the Titanic dataset in zip format.

2. Update the `df_zip` variable with the correct path to your zip file.

3. Run the `titanic_survival_prediction.py` script.

## Data Preprocessing

- The code starts by unzipping the Titanic dataset and reading the 'train.csv' file using pandas.

- It performs data preprocessing, including handling missing values, extracting and cleaning titles, and encoding categorical features.

## Model Building

- The code builds a deep learning model using Keras. It consists of an input layer, hidden layers with dropout, and an output layer.

- The model is compiled with the Adam optimizer, binary cross-entropy loss, and accuracy as a metric.

## Training and Visualization

- The training and validation process is performed for 50 epochs.

- The training history (accuracy) is visualized using matplotlib.

## Contributing

Contributions are welcome. If you have any suggestions or improvements, please create a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

