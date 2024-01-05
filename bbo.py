import nevergrad as ng
from concurrent import futures
import os
import time


import numpy as np
from matplotlib import pyplot as plt


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
def pca_online(h:float, alpha:float) -> float:

    global nominal_dataset
    global anomalous_dataset
    command = "./anomaly_detector n {h} {alpha} n y {dataset}"

    w1 = 0.4
    w2 = 0.6

    # Check False Acceptance Rate
    output = os.popen(command.format(h=h, alpha=alpha, dataset=nominal_dataset)).readlines()[0]
    FAR = float(output)

    # Check False Rejection Rate
    output = os.popen(command.format(h=h, alpha=alpha, dataset=anomalous_dataset)).readlines()[0]
    FRR = 1 - float(output)

    loss = w1*FAR + w2*FRR
    
    global history
    history.append((loss, h, alpha))
    return loss


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
def gem_online(h:float, alpha:float) -> float:

    global nominal_dataset
    global anomalous_dataset
    command = "./anomaly_detector y {h} {alpha} n y {dataset}"

    w1 = 0.4
    w2 = 0.6

    # Check False Acceptance Rate
    output = os.popen(command.format(h=h, alpha=alpha, dataset=nominal_dataset)).readlines()[0]
    FAR = float(output)

    # Check False Rejection Rate
    output = os.popen(command.format(h=h, alpha=alpha, dataset=anomalous_dataset)).readlines()[0]
    FRR = 1 - float(output)

    loss = w1*FAR + w2*FRR
    
    global history
    history.append((loss, h, alpha))
    return loss


'''
Run the Black-Box Optimizer
'''
model             = "gem"
optimizer_name    = "NGOpt"
nominal_dataset   = "./datasets/nominal-human-activity.csv" # safe samples
anomalous_dataset = "./datasets/anomaly-human-activity.csv" # anomalous samples
 
num_workers       = 4
num_iterations    = 125 * num_workers # write it as budget-per-worker

# generate test script
os.system("bash ./get_test_datasets.sh")

#      h     alpha
lb = [ 1.0,  0.01] # lower-bound
ub = [10.0,  1.00] # upper bound
instrumentation = ng.p.Instrumentation(
	h = ng.p.Log(lower=lb[0], upper=ub[0]),
	alpha = ng.p.Log(lower=lb[1], upper=ub[1])
)


optimizer = ng.optimizers.registry[optimizer_name](instrumentation, budget=num_iterations, num_workers=num_workers)

# do the job
history = []
start_time = time.time()
print("Begin offline Phase")

if model == "pca":
    pca_offline() 
    print("Begin Black-Box Optimization")
    # Better use Thread or Process? TODO: check Nevergrad/futures docs, my guess is threads
    with futures.ThreadPoolExecutor(max_workers=optimizer.num_workers) as executor:
        recommendation = optimizer.minimize(pca_online, executor=executor, batch_mode=False, verbosity=1)

elif model == "gem":
    gem_offline() 
    print("Begin Black-Box Optimization")
    with futures.ThreadPoolExecutor(max_workers=optimizer.num_workers) as executor:
        recommendation = optimizer.minimize(gem_online, executor=executor, batch_mode=False, verbosity=1)

else:
    os._exit(1)

# clean up datasets
os.system("bash ./remove_test_datasets.sh")
os.system("bash ./remove_garbage.sh")

# save log
res_string = "--- BBO took {time} seconds ---".format(time=(time.time()-start_time))
res_string += "\nModel: {model}".format(model=model)
res_string += "\nOffline phase: {nds}".format(nds=nominal_dataset)
res_string += "\nOnline phase:  {ads}".format(ads=anomalous_dataset)
res_string += "\nOptimizer: {opt}".format(opt=optimizer_name)
res_string += "\nParameters: " + str(recommendation.kwargs) + "\nBest value: " + str(recommendation.loss) + '\n'

print(res_string)

with open ("./log.txt", 'a') as f:
    f.write(res_string)


# plot the super graph
history.sort(key=lambda x: x[0])
L, H, A = zip(*history)

ax = plt.figure().add_subplot(projection='3d')
ax.plot(H, A, L, label='{model} Optimization Curve'.format(model=model))

ax.set_xlabel('h') 
ax.set_ylabel('alpha') 
ax.set_zlabel('Loss') 

plt.savefig('{model}_bbo_3d_graph.png'.format(model=model))
plt.show()
