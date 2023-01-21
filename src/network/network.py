from keras import models
from keras import layers
from keras import optimizers
from keras import losses
from keras import metrics

#creating the network consisting of two hidden layers
model = models.Sequential()
model.add(layers.Dense(16, activation = 'relu', input_shape = (10000,)))
model.add(layers.Dense(16, activation = 'relu'))
#"probability" that the movie has a positive review
model.add(layers.Dense(1, activation = 'sigmoid'))

#since this is a binary classification problem, we will use 
#binary cross entropy
#RMS prop with learning rate 0.001
model.compile(optimizer= optimizers.RMSprop(learning_rate = 0.001),
              loss = losses.binary_crossentropy,
              metrics = [metrics.binary_accuracy])

              




