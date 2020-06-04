# -*- coding: utf-8 -*-

"""
训练cnews模型
"""

from model import *
from data_helper import *
import matplotlib.pyplot as plt


max_seq_len=200
vocab_size=6000
embed_size=300
checkpoint_save_path = "./checkpoint/model.ckpt"
batch_size=128
epochs=10


train_data = "./data/cnews/cnews.train.txt"
test_data = "./data/cnews/cnews.test.txt"
dev_data = "./data/cnews/cnews.val.txt"
vocab_data = "./data/cnews/cnews.vocab.txt"
word2id_path = "./data/word2id.json"
label2id_path = "./data/cnews/label2id.json"

# 加载cnews数据
x_train, y_train, x_test, y_test, x_dev, y_dev = transfor_cnews(train_data, dev_data, test_data,
                                                                vocab_data, word2id_path, label2id_path, max_seq_len)


model = TextCnn(vocab_size=vocab_size, embed_size=embed_size)

model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['sparse_categorical_accuracy'])

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                 save_weights_only=False,
                                                 save_best_only=True,
                                                 monitor='val_loss',
                                                 verbose=1)

history = model.fit(x_train, y_train,
                    validation_data=(x_dev, y_dev),
                    batch_size=batch_size, epochs=epochs,  callbacks=[cp_callback])

model.evaluate(x_test, y_test)

model.summary()



# 显示训练集和验证集的acc和loss曲线
acc = history.history['sparse_categorical_accuracy']
val_acc = history.history['val_sparse_categorical_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
# plt.show()
plt.savefig("./train.png")