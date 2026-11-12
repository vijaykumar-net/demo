# program 1 deep forward ANN using back propagation using a sequential dataset

from keras.models import Sequential
from keras.layers import Dense, Activation
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


X, y = load_iris(return_X_y = True)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.4)


model = Sequential()


for i in range(3):
  model.add(Dense(2, input_shape = (4,)))
  model.add(Activation('sigmoid'))
  model.add(Dense(1))
  model.add(Activation('sigmoid'))


model.compile(
    loss= 'mean_squared_error',
    optimizer='sgd',
    metrics=['accuracy']
)


model.summary()
model.fit(X_train, y_train,epochs=5)
score = model.evaluate(X_test,y_test)
print(score)
