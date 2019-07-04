##CNN----classifying images as cat,dog,human,table,building et


from keras.models import Sequential  ##to initializing a NN..layers or graph
                                    
from keras.layers import Convolution2D    #step-1
from keras.layers import MaxPooling2D     #step-2
from keras.layers import Flatten          #step-3 flattening
from keras.layers import Dense            #to add fully connected layer/s in CNN

#using tensorflow in backend...what is tensorflow..

classifier =Sequential()   ###initiallized CNN

##Convolution::
##using many filters(=> many feature maps as output of this step-1)

classifier.add(Convolution2D(32,(3,3),padding='valid',input_shape=(64,64,3),activation='relu')) #64 feature detectors,3x3 filters
                                                            #input_shape(1 for B/W images 3 for colored//rgb..)
                                                            #now the size of the image as input 64x64
                                                            ##activation function
                     
#now we have 5x5 feature map                                       
                                               
#step-2::MAX_POOLING..
    ##5x5 to 3x3::

classifier.add(MaxPooling2D(pool_size=(2,2)))        ##2x2 pool size to get the max in..
                                                                    ##we wanna preserve the edge information too..sometimes very important                                               


#step-3:  why we are flattenning we wont loss spatial info doing that we dont because we used filters
        #to get some very high values like 4 out of 1 and 0's,so to put these into input to an ann
        ##we are going to flatten them into a vector::
        ##//high numbers represents spatial info nicely
        
        
classifier.add(Flatten())


##step-4:   Full connection step     ##hidden layer addition::
            #    just put this classifier object to the input of the ann..
        
        
        
classifier.add(Dense(128,activation='relu'))
                                            #output_dim=number of hidden neurons
    
##output layer::
                                            
classifier.add(Dense(1,activation='sigmoid'))

                                                        ##use soft_max for more than 2 categories
                                                       #output_dim=number of output neurons

    
    
    
##compiling the CNN:
                                                      
                                                        
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
                                                    ##adam for stocastic grad descent
                                                    #binary for cat and dog categorical for more than 2
    
    
  
 ##image fit into the above CNN built::##to save from overfitting  on the training set..//keras image augmenetation

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_set = train_datagen.flow_from_directory('E:\pers_proj1\dataset',target_size=(64,64),
                                                 batch_size=32,class_mode='binary')##binary _outcome..cat or dog

test_set = test_datagen.flow_from_directory(
        'E:\pers_pro1b',
        target_size=(64,64),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(
        train_set,
        steps_per_epoch=8000,
        epochs=1,
        validation_data=test_set,
        validation_steps=2000)    
    
                                                                    
                                                                





