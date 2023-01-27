import numpy as np
from tensorflow import keras
from src.util.load_review import return_tensored_review

#cutoff value for the sentiment
cutoff = 0.5

#loading movie review
movie_review = return_tensored_review("pale_blue_eyes.txt")

#loading the saved model
model = keras.models.load_model("src/saved_model")


prediction = model.predict(movie_review.reshape(1,10000))
if prediction[0] < cutoff:
   print("negative")
if prediction[0] > cutoff:
   print("positive")

