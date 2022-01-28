# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 01:37:23 2022

@author: A
"""

#%%
head_ft = tf.keras.models.clone_model(model_pfe)
head_ft.set_weights(model_pfe.get_weights())

base_model_ft = tf.keras.applications.efficientnet.EfficientNetB7(
    input_shape=(32, 32, 3), # refers to the shape we transfer from
    include_top=False,       # cut off the head
    weights='imagenet',      # pretrained on the ImageNet data
)
base_model_ft.training = False # to disable updating of means and 
                               # variances in batch norm layers
                               # This is in some cases very important, 
                               # but in this case it would still 
                               # work fine if we did not do it

for layer in base_model_ft.layers[:100]: # layers after number 100 
    layer.trainable = False              #in base_model now trainable
    
model_ft = tf.keras.models.Sequential([
    data_augmentation, # can still use data augmentation now
    base_model_ft, # the pre-trained part
    head_ft, # the classifer we trained using feature extraction
])

model_ft.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), # use a low learning rate
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
    )
#%%
model_ft.summary()
#%%
history_ft = model_ft.fit(2 * train_images - 1, train_labels, 
                          validation_data=(2 * test_images - 1, test_labels), 
                          epochs=15, verbose=1)