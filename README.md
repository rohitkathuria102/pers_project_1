# pers_project_1
CV_project_1
This is my pers project-1 of classification of various images in 4 classes// CAT,CAR,HUMAN,DOG..



2 Convolution + Pooling layers::
3#64x64 ===> 3@(128@(15x15))

Convolution step(with one 3x3 filter)::==> 3@(128@(11x11))
Max_pooling step (with 2x2 pool_size)::==> 3@(128@(6x6))

=> flattening:: 3@(128 samples of 36 vector size)
i.e,
128 vectors of r image ---with vector size of 36
128 vectors of g image ---with vector size of 36
128 vectors of b image ---with vector size of 36

FULLY CONNECTED LAYER(HIDDEN)::

from keras.layers import Dense
classifier1.add(Dense(128,activation='relu'))  // 128 neurons in the hidden layer..

==>
OUTPUT LAYER::

classifier.add(Dense(1,activation='softmax'))


model.compile( loss='categorical_crossentropy', optimizer=keras.optimizers.SGD(), metrics=[keras.metrics.categorical_accuracy])

