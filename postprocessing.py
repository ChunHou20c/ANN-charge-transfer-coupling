# these file contain commonly used function for processing the data
import matplotlib.pyplot as plt
import config
from sklearn.metrics import r2_score
import numpy as np

def epoches_graph(Dictionary):

    keys = list(Dictionary.keys())
    loss = keys[0]
    metrics = keys[1]
    val_loss = keys[2]
    val_metrics = keys[3]
    _epoches = [i for i in range(len(Dictionary[loss]))]

    plt.plot(_epoches, Dictionary[loss], '--r',label=config.LOSS)
    plt.plot(_epoches, Dictionary[val_loss], 'g',label="validation "+config.LOSS)
    plt.legend()
    plt.title(config.LOSS +" vs epoches")
    plt.xlabel("epoches")
    plt.annotate( config.LOSS + ":\n\ntraining = {:.5f}".format(Dictionary[loss][-1]) + 
        "\nvalidation = {:.5f}".format(Dictionary[val_loss][-1]), xy = (0.3,0.3), xycoords="figure fraction")
    plt.ylabel(config.LOSS)
    plt.yscale("log")
    plt.savefig(config.SAVE_PATH+ "/loss vs epoches.png")
    plt.clf()

    
    plt.plot(_epoches, Dictionary[metrics], '--r',label=config.METRICS)
    plt.plot(_epoches, Dictionary[val_metrics], 'g',label="validation " + config.METRICS)
    plt.legend()
    plt.title(config.METRICS + " vs epoches")
    plt.xlabel("epoches")
    plt.annotate(config.METRICS + ":\n\ntraining = {:.5f}".format(Dictionary[metrics][-1]) + 
        "\nvalidation = {:.5f}".format(Dictionary[val_metrics][-1]), xy = (0.3,0.3), xycoords="figure fraction")
    plt.ylabel(config.METRICS)
    plt.yscale("log")
    plt.savefig(config.SAVE_PATH + "/metrics vs epoches.png")
    plt.clf()

def save_scatter_graph(_predicted_y, _actual_y, _title):
    
    _predicted_y = _predicted_y.flatten()
    _actual_y = _actual_y.flatten()
    arr = np.concatenate((_predicted_y,_actual_y),axis=0)
    y_max = max(arr)
    y_min = min(arr)
    diagonal = [y_max, y_min]
    plt.plot(diagonal,diagonal,'b')
    plt.scatter(_predicted_y,_actual_y)
    plt.annotate("r-squared = {:.3f}".format(r2_score(_predicted_y, _actual_y)), xy=(0.6,0.3),xycoords='figure fraction')
    plt.title(_title)
    plt.ylabel("Actual Y")
    plt.xlabel("Predicted Y") #+'('+"r-squared = {:.3f}".format(r2_score(_actual_y, _predicted_y))+')')
    plt.savefig(config.SAVE_PATH + '/' + _title + ".png")
    plt.clf()
