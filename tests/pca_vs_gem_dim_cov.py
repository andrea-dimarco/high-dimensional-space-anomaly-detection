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
p = 1       # data dimension
N1 = 1000    # Number of samples to generate for training
N2 = N1 * 2 # Nmber of samples to generate for testing


gem_alpha = 0.4
gem_h = 5
pca_alpha = gem_alpha
pca_h = gem_h

delta = 1
num_trials = 1
num_iterations = 500

mean = 0.0
noise = 0.2 # for the perturbation

# files
file_path = "./datasets/"
nominal_dataset = file_path + "exp_5_train.csv"
anomalous_dataset = file_path + "exp_5_test.csv"


# do the experiments
start_time = time.time()
pca_anomaly_history = []
gem_anomaly_history = []
dimension_history = []
generate_dataset = "./generate_dataset n {mean} {variance} {dim} {samples} {path}"
for i in range(num_iterations):

    cov_1 = np.eye(p)
    mu = np.zeros(p)

    # test dataset
    training_data = np.random.multivariate_normal(mean=mu, cov=cov_1, size=N2).transpose()
    # save datasets in csv files
    df = pd.DataFrame(training_data)
    df.to_csv(anomalous_dataset, index=False, header=False)

    # train dataset
    noise_matrix = np.random.uniform(0,noise,size=(p,p))
    noise_matrix = (noise_matrix + noise_matrix.T) / 2
    np.fill_diagonal(noise_matrix,0)
    noisy_cov = cov_1 + noise_matrix
    df = pd.DataFrame(np.random.multivariate_normal(mean=mu, cov=noisy_cov, size=N1).transpose())
    df.to_csv(nominal_dataset, index=False, header=False)

    # run models
    pca_offline()
    pca_anomaly_rate = pca_online(pca_h, pca_alpha, num_trials)
    gem_offline()
    gem_anomaly_rate = gem_online(gem_h, gem_alpha, num_trials)

    pca_anomaly_history.append(pca_anomaly_rate)
    gem_anomaly_history.append(gem_anomaly_rate)
    dimension_history.append(p)

    print("{i}: gem={gem} pca={pca}".format(i=i, gem=gem_anomaly_rate, pca=pca_anomaly_rate))
    
    p += delta

# save log
res_string = "--- PCA vs GEM (dimension) took {time} seconds ---".format(time=(time.time()-start_time))
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
plt.title('Anomalies with increasing dimensions')
  
# function to show the plot 
plt.legend() 
plt.savefig("pca_vs_gem_dim_cov.png")
plt.show()

os.system("bash ./tests/remove_garbage.sh")
os.system("bash ./tests/remove_test_datasets.sh")

os._exit(0)
