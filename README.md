# Assignment (Homework) #2
> Artur Khayaliev, BS4-DS
### Part 1: Deep Learning
  1. Build a DNN with five hidden layers of 100 neurons each, He initialization, and the ELU
  activation function.
  2. Using Adam optimization and early stopping, try training it on MNIST but only on digits 0
  to 4, as we will use transfer learning for digits 5 to 9 in the next exercise. You will need a
  softmax output layer with five neurons.
  3. Tune the hyperparameters using cross-validation and see what precision you can
  achieve.
  4. Now try adding Batch Normalization and compare the learning curves: is it converging
  faster than before? Does it produce a better model?
  5. Is the model overfitting the training set? Try adding dropout to every layer and try
  again. Does it help?
### Part 2: Transfer learning
  1. Create a new DNN that reuses all the pretrained hidden layers of the previous model,
  freezes them, and replaces the softmax output layer with a fresh new one.
  2. Train this new DNN on digits 5 to 9, using only 100 images per digit, and time how long
  it takes. Despite this small number of examples, can you achieve high precision?
  3. Try caching the frozen layers, and train the model again: how much faster is it now?
  4. Try again reusing just four hidden layers instead of five. Can you achieve a higher
  precision?
  5. Now unfreeze the top two hidden layers and continue training: can you get the model
  to perform even better?
