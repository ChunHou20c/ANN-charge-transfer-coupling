import config
import model
import preprocessing
import postprocessing
import tensorflow as tf
import callbacks

def main():
    Model = model.model
    Model.summary()
    Model.compile(optimizer=config.OPTIMIZER,
        loss = config.LOSS, 
        metrics = config.METRICS)

    history = Model.fit(preprocessing.x_train,
        preprocessing.y_train,
        batch_size = config.BATCH_SIZE,
        epochs = config.EPOCHES,
        validation_data = (preprocessing.x_val, preprocessing.y_val),
        callbacks=callbacks.CALLBACKS)

    postprocessing.epoches_graph(history.history)

    Model.evaluate(preprocessing.x_train,preprocessing.y_train)
    Model.evaluate(preprocessing.x_val,preprocessing.y_val)
    Model.evaluate(preprocessing.x_test,preprocessing.y_test)


    predicted_y_train_set = Model.predict(preprocessing.x_train)
    postprocessing.save_scatter_graph(predicted_y_train_set, preprocessing.y_train, "Y vs predicted Y (train set)")

    predicted_y_val_set = Model.predict(preprocessing.x_val)
    postprocessing.save_scatter_graph(predicted_y_val_set, preprocessing.y_val, "Y vs predicted Y (val set)")

    if (config.USE_TEST_SET == True):
        predicted_y_test_set = Model.predict(preprocessing.x_test)
        postprocessing.save_scatter_graph(predicted_y_test_set, preprocessing.y_test, "Y vs predicted Y (test set)")

if __name__ == '__main__':
    main()
