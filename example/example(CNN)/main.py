import model
import preprocessing
import config
import postprocessing
import tensorflow as tf
from schedular import lr_exp_decay as scheduler

def main():
    Model = model.model
    Model.compile(optimizer=config.OPTIMIZER,
        loss = config.LOSS, 
        metrics = config.METRICS)

    callback = tf.keras.callbacks.LearningRateScheduler(scheduler)


    history = Model.fit(preprocessing.x_train,
        preprocessing.y_train,
        batch_size = config.BATCH_SIZE,
        epochs = config.EPOCHES,
        validation_data = (preprocessing.x_val, preprocessing.y_val),
        callbacks=[callback])

    predicted_y_test_set = Model.predict(preprocessing.x_test)
    predicted_y_val_set = Model.predict(preprocessing.x_val)
    predicted_y_train_set = Model.predict(preprocessing.x_train)

    postprocessing.epoches_graph(history.history)
    postprocessing.save_scatter_graph(predicted_y_test_set, preprocessing.y_test, "Y vs predicted Y (test set)")
    postprocessing.save_scatter_graph(predicted_y_val_set, preprocessing.y_val, "Y vs predicted Y (val set)")
    postprocessing.save_scatter_graph(predicted_y_train_set, preprocessing.y_train, "Y vs predicted Y (train set)")

if __name__ == '__main__':
    main()
