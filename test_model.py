import keras
from src.imdb_dataset.prepare_data import x_test
from src.imdb_dataset.prepare_data import y_test

#importing saved model
model = keras.models.load_model('src/saved_model')
results = model.evaluate(x_test,y_test)

if __name__ == "__main__":
    print(results)
