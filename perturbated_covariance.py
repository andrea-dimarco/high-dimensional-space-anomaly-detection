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
    print("PCA offline phase done.")

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
    print("GEM offline phase done.")

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
p = 10       # data dimension
N1 = 1000    # Number of samples to generate for training
N2 = N1 * 2 # Nnmber of samples to generate for testing
model = "pca"

alpha = 0.1
h = 2

offset = 0.1
delta = 0.1
num_trials = 2
num_iterations = 500

# mean vector
mean = 0.0
mu = np.zeros(p)

# covariance matrixes
cov_1 = np.eye(p)


# files
file_path = "./datasets/"
nominal_dataset = file_path + "exp_2_train.csv"
anomalous_dataset = file_path + "exp_2_test.csv"

# Generate samples from multivariate normal distribution
training_data = np.random.multivariate_normal(mean=mu, cov=cov_1, size=N1).transpose()

# save datasets in csv files
df = pd.DataFrame(training_data)
df.to_csv(nominal_dataset, index=False, header=False)
del training_data

# train model
if model == "pca":
    pca_offline()
else:
    gem_offline()

# do the experiments
start_time = time.time()
anomaly_history = []
offset_history = []
for i in range(num_iterations):
    # save dataset
    cov_2 = cov_1 + np.random.uniform(mean-offset, mean+offset, (p,p))
    df = pd.DataFrame(np.random.multivariate_normal(mean=mu, cov=cov_2, size=N2).transpose())
    df.to_csv(anomalous_dataset, index=False, header=False)

    # run model
    if model == "pca":
        anomaly_rate = pca_online(h, alpha, num_trials)
    else:
        anomaly_rate = gem_online(h, alpha, num_trials)

    anomalies_found = int(N2 * anomaly_rate)
    anomaly_history.append(anomalies_found)
    offset_history.append(offset)
    print(str(i+1) + ": " + str(anomalies_found) + " anomalies found for offset = " + str(round(offset, 2)))
    offset += delta

# save log
res_string = "--- Perturbation took {time} seconds ---".format(time=(time.time()-start_time))
res_string += "\nOffline phase: {nds}".format(nds=nominal_dataset)
res_string += "\nOnline phase:  {ads}\n".format(ads=anomalous_dataset)

print(res_string)

with open ("./log.txt", 'a') as f:
    f.write(res_string)

# plotting the points  
plt.plot(offset_history, anomaly_history) 
  
# naming the x axis 
plt.xlabel('Covariance perturbation') 
# naming the y axis 
plt.ylabel('Anomalies found') 
  
# giving a title to my graph 
plt.title('{model} anomalies by perturbing the covariance'.format(model=model)) 
  
# function to show the plot 
plt.savefig("{model}_perturbated_covariance.png".format(model=model))
plt.show()