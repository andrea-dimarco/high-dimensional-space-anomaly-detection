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

k = 1
delta = 0.01
num_trials = 5
num_iterations = 500

# mean vector
mean = 0.0
mu = np.zeros(p)

# covariance matrixes
cov_1 = np.eye(p)

# files
file_path = "./datasets/"
nominal_dataset = file_path + "exp_1_train.csv"
anomalous_dataset = file_path + "exp_1_test.csv"

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
k_history = []
for i in range(num_iterations):
    # save dataset
    df = pd.DataFrame(np.random.multivariate_normal(mean=mu, cov=k*cov_1, size=N2).transpose())
    df.to_csv(anomalous_dataset, index=False, header=False)
    # run model
    if model == "pca":
        anomaly_rate = pca_online(h, alpha, num_trials)
    else:
        anomaly_rate = gem_online(h, alpha, num_trials)

    anomalies_found = int(N2 * anomaly_rate)
    anomaly_history.append(anomaly_rate)
    k_history.append(delta*i)
    print(str(i) + ": " + str(anomalies_found) + " anomalies found for k = " + str(round(k, 2)))
    k += delta


# plotting the points  
plt.plot(k_history, anomaly_history) 
  
# naming the x axis 
plt.xlabel('Covariance offset') 
# naming the y axis 
plt.ylabel('Anomalies found') 
  
# giving a title to my graph 
plt.title('{model} anomalies with increasing variance'.format(model=model)) 
  
# function to show the plot 
plt.savefig("{model}_linear_scale_covariance.png".format(model=model))
plt.show()

# visualize result
print("\n--- %s seconds ---" % (time.time() - start_time))
print("Model:", model)
print("Offline phase:", nominal_dataset)
print("Online phase: ", anomalous_dataset)
os._exit(0)