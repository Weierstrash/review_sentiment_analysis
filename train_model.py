from src.imdb_dataset.prepare_data import x_train
from src.imdb_dataset.prepare_data import y_train
from src.network.network import model
import keras
import matplotlib.pyplot as plt

#creating a validation set
#by setting aside 10000 samples from 
#the original 60000 samples
x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

epochs = 4
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs = epochs,
                    batch_size = 512,
                    validation_data = (x_val,y_val))

#saving the model
keras.models.save_model(model,'src/saved_model/')



#plotting the training/validation accuracy
history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']

acc = history_dict.values()
acc = list(acc)
acc = acc[0]

epochs = range(1,len(acc)+1)
plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()



