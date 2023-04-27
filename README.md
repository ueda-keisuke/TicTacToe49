# Tic-Tac-Toe AlphaZero

This project is a restructured version created for study purposes, based on "AlphaZero that even a 49-year-old dude could make this year (https://github.com/tail-island/tictactoe-ai)." While there are differences in the details, please note that it was created using the original code as a foundation.

As there is no explicit license in the original project, this project will be licensed under an open-source license, specifically the MIT License.

This repository contains an implementation of the AlphaZero algorithm for the game of Tic-Tac-Toe, using Keras to build and train the model.
## Requirements

* Keras
* numpy
* pathlib 
* pickle
* datetime

## How it works

The project is composed of three main parts:

1. **Generate training data**: Generate self-play data to be used for training.
2. **Train the model**: Train the model using the generated training data.
3. **Evaluate the model**: Compare the performance of the new model with the current champion.

## Generate Training Data

The script `generate_data.py` generates self-play data by simulating games of Tic-Tac-Toe using the latest model. It plays a given number of games (`MAX_GAME_COUNT`) and stores the game states and corresponding action probabilities in pickle files.
## Train the Model

The script `train.py` trains the model using the most recent training data generated. It uses a learning rate scheduler to gradually decrease the learning rate during training. After training, the updated model is saved as a new candidate model.
## Evaluate the Model

The script evaluate.py evaluates the new candidate model against the current champion model by simulating games between them. If the candidate model achieves a win rate higher than a specified threshold, it replaces the current champion.
## Usage

1. Run `init_model.py` to initialize the model.
2. Run `generate_data.py` to generate training data.
3. Run `train.py` to train the model using the generated data.
4. Run `evaluate.py` to evaluate the new candidate model against the current champion model.

## Model Architecture

The model architecture consists of a series of residual blocks, followed by global average pooling and two dense output layers: one for the policy and one for the value.
##Customization

You can adjust the following parameters to suit your needs:

* MAX_GAME_COUNT: The number of games played to generate training data.
* MCTS_EVALUATE_COUNT: The number of MCTS evaluations performed for each move.
* TEMPERATURE: A parameter controlling the exploration/exploitation trade-off during MCTS.

Please note that these parameters may affect the performance of the algorithm and should be tuned accordingly.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.
## Contributing

If you would like to contribute to this project, please feel free to fork the repository, make your changes, and submit a pull request. We appreciate your help!
## Acknowledgements

This project is a restructured version created for study purposes, based on "AlphaZero that even a 49-year-old dude could make this year (https://github.com/tail-island/tictactoe-ai)." While there are differences in the details, please note that it was created using the original code as a foundation.

This project is inspired by the original AlphaZero paper by DeepMind:

* Silver, D., Hubert, T., Schrittwieser, J., Antonoglou, I., Lai, M., Guez, A., ... & Hassabis, D. (2018). A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play. Science, 362(6419), 1140-1144.

Additionally, we would like to thank the open-source community for providing valuable resources and examples to help develop this project.
