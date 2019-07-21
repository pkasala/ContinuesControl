import matplotlib.pyplot as plt

import numpy as np
from datetime import datetime

directory="./results/"
date_sufix = ""

#save the progress of success
def save_plot_scores(scores,filename):
    # plot the scores
    fig = plt.figure()
    fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    #save the plot to file
    plt.savefig(filename + '.png')
    plt.close(fig)


#prepare the file name for model
def model_file_name(config):
    date_sufix = datetime.now().strftime("%m_%d_%Y_%H_%M")
    file_name = "%s_%s_%s" % (config.model_name, 'model',date_sufix)
    # get date for storing results
    file_name = directory+"%s" % (file_name)
    return  file_name

#prepare the file name for plot statistics score
def plot_file_name(config):
    date_sufix = datetime.now().strftime("%m_%d_%Y_%H_%M")
    file_name = "%s_%s_%s" % (config.model_name, 'plot',date_sufix)
    # get date for storing results
    file_name = directory+"%s" % (file_name)
    return  file_name
