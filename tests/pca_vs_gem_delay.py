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
def pca_online_delay(h:float, alpha:float) -> float:

    global anomalous_dataset
    command = "./anomaly_delay n {h} {alpha} n y {dataset}"

    # Run simulation
    delay = int(os.popen(command.format(h=h, alpha=alpha, dataset=anomalous_dataset)).readlines()[0])

    return delay


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
def gem_online_delay(h:float, alpha:float) -> float:

    global anomalous_dataset
    command = "./anomaly_delay y {h} {alpha} n y {dataset}"

    # Run simulation
    output = os.popen(command.format(h=h, alpha=alpha, dataset=anomalous_dataset)).readlines()[0]
    delay = int(output)

    return delay


# parameters
p = 100       # data dimension
N1 = 1000    # Number of samples to generate for training
N2 = N1 * 2 # Nmber of samples to generate for testing

gem_alpha = 0.4
gem_h = 5
pca_alpha = gem_alpha
pca_h = gem_h

k = 1
delta = 0.001
num_trials = 1
num_iterations = 100

# mean vector
mean = 0.0
mu = np.zeros(p)

# covariance matrixes
cov_1 = np.eye(p)

# files
file_path = "./datasets/"
nominal_dataset = file_path + "exp_train.csv"
anomalous_dataset = file_path + "exp_test.csv"

# Generate samples from multivariate normal distribution
training_data = np.random.multivariate_normal(mean=mu, cov=cov_1, size=N1).transpose()

# save datasets in csv files
df = pd.DataFrame(training_data)
df.to_csv(nominal_dataset, index=False, header=False)
del training_data

# do the experiments
start_time = time.time()
pca_delay_history = []
gem_delay_history = []
k_history = []
pca_offline()
gem_offline()

pca_found = False
gem_found = False

for i in range(num_iterations):
    # save dataset
    df = pd.DataFrame(np.random.multivariate_normal(mean=mu, cov=k*cov_1, size=N2).transpose())
    df.to_csv(anomalous_dataset, index=False, header=False)

    # run model
    gem_delay = gem_online_delay(gem_h, gem_alpha)
    pca_delay = pca_online_delay(pca_h, pca_alpha)

    pca_delay_history.append(pca_delay)
    gem_delay_history.append(gem_delay)

    k_history.append(delta*i)
    
    print("{i}: gem={gem} pca={pca}".format(i=i, gem=gem_delay, pca=pca_delay))

    k += delta


# save log
res_string = "--- Delay test took {time} seconds ---".format(time=(time.time()-start_time))
res_string += "\nOffline phase: {nds}".format(nds=nominal_dataset)
res_string += "\nOnline phase:  {ads}\n".format(ads=anomalous_dataset)

print(res_string)

with open ("./log.txt", 'a') as f:
    f.write(res_string)



# plotting the points  
plt.plot(k_history, pca_delay_history, label="PCA") 
plt.plot(k_history, gem_delay_history, label="GEM") 
  
# naming the x axis 
plt.xlabel('Variance') 
# naming the y axis 
plt.ylabel('Delay') 
  
# giving a title to my graph 
plt.title('Delay with increasing variance')
  
# function to show the plot 
plt.legend() 
plt.savefig("pca_vs_gem_delay.png")
plt.show()

os.system("bash ./tests/remove_garbage.sh")
os.system("bash ./tests/remove_test_datasets.sh")

os._exit(0)
