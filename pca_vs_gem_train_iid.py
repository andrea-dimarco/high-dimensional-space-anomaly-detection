import numpy as np
import pandas as pd
import os
import time
import matplotlib.pyplot as plt 

'''
This function performs the offline phase of the PCA model
'''
def pca_offline():
    global nominal_dataset
    command = "./anomaly_detector n 1.0 1.0 y y {dataset}"
    os.system(command.format(dataset=nominal_dataset))

'''
This function will call the PCA model and returns the loss value
'''
def pca_online(h:float, alpha:float, num_trials:int) -> float:

    global anomalous_dataset
    command = "./anomaly_detector n {h} {alpha} n y {dataset}"

    # Run simulation
    output = 0
    for i in range(num_trials):
        output = output + float(os.popen(command.format(h=h, alpha=alpha, dataset=anomalous_dataset)).readlines()[0])
    
    res = output / float(num_trials)

    return res


'''
This function performs the offline phase of the GEM model
'''
def gem_offline():
    global nominal_dataset
    command = "./anomaly_detector y 1.0 1.0 y y {dataset}"
    os.system(command.format(dataset=nominal_dataset))

'''
This function will call the PCA model and returns the loss value
'''
def gem_online(h:float, alpha:float, num_trials:int) -> float:

    global anomalous_dataset
    command = "./anomaly_detector y {h} {alpha} n y {dataset}"

    # Run simulation
    output = 0
    for i in range(num_trials):
        output = output + float(os.popen(command.format(h=h, alpha=alpha, dataset=anomalous_dataset)).readlines()[0])
    
    res = output / float(num_trials)

    return res


# parameters
p = 100     # data dimension
N1 = 2      # Number of samples to generate for training
N2 = 1000   # Nnmber of samples to generate for testing

nominal_mean = 0.0
nominal_variance = 1.0
anomalous_mean = 0.5
anomalous_variance = 1.5

gem_alpha = 0.5
gem_h = 8
pca_alpha = gem_alpha
pca_h = gem_h

delta = 1
num_trials = 1
num_iterations = 1000

# files
file_path = "./datasets/"
nominal_dataset = file_path + "exp_4_train.csv"
anomalous_dataset = file_path + "exp_4_test.csv"

# do the experiments
start_time = time.time()
pca_anomaly_history = []
gem_anomaly_history = []
dimension_history = []
generate_dataset = "./generate_dataset n {mean} {variance} {dim} {samples} {path}"
for i in range(num_iterations):
    # train dataset
    os.system(generate_dataset.format(mean=nominal_mean, variance=nominal_variance, dim=p, samples=N1, path=nominal_dataset))
    # test dataset
    os.system(generate_dataset.format(mean=anomalous_mean, variance=anomalous_variance, dim=p, samples=N2, path=anomalous_dataset))

    # run models
    pca_offline()
    pca_anomaly_rate = pca_online(pca_h, pca_alpha, num_trials)
    gem_offline()
    gem_anomaly_rate = gem_online(gem_h, gem_alpha, num_trials)

    pca_anomaly_history.append(pca_anomaly_rate)
    gem_anomaly_history.append(gem_anomaly_rate)
    dimension_history.append(N1)

    print("{i}: gem={gem} pca={pca}".format(i=i, gem=gem_anomaly_rate, pca=pca_anomaly_rate))

    N1 += delta

# save log
res_string = "--- PCA vs GEM (train) took {time} seconds ---".format(time=(time.time()-start_time))
res_string += "\nOffline phase: {nds}".format(nds=nominal_dataset)
res_string += "\nOnline phase:  {ads}\n".format(ads=anomalous_dataset)

print(res_string)

with open ("./log.txt", 'a') as f:
    f.write(res_string)

# plotting the points  
plt.plot(dimension_history, pca_anomaly_history, label="PCA") 
plt.plot(dimension_history, gem_anomaly_history, label="GEM") 
  
# naming the x axis 
plt.xlabel('Dimension') 
# naming the y axis 
plt.ylabel('Anomalies found') 
  
# giving a title to my graph 
plt.title('Anomalies with increasing training samples')
  
# function to show the plot 
plt.legend() 
plt.savefig("pca_vs_gem_train.png")
plt.show()

os._exit(0)