import nevergrad as ng
from concurrent import futures
import os
from subprocess import run

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

    # Check False Acceptance Rate
    output = os.popen(command.format(h=h, alpha=alpha, dataset=nominal_dataset)).readlines()[0]
    FAR = float(output)

    # Check False Rejection Rate
    output = os.popen(command.format(h=h, alpha=alpha, dataset=anomalous_dataset)).readlines()[0]
    FRR = 1 - float(output)
    
    return FAR + FRR


'''
This function performs the offline phase of the GEM model
'''
def gem_offline():
    global nominal_dataset
    command = "./anomaly_detector y 1.0 1.0 n y {dataset}"
    os.system(command.format(dataset=nominal_dataset))
    print("PCA offline phase done.")

'''
This function will call the PCA model and returns the loss value
'''
def gem_online(h:float, alpha:float) -> float:

    global nominal_dataset
    global anomalous_dataset
    command = "./anomaly_detector y {h} {alpha} n y {dataset}"

    # Check False Acceptance Rate
    os.system(command.format(h=h, alpha=alpha, dataset=nominal_dataset))
    with open ("loss.txt", 'r') as f:
        FAR = float(f.readline())
        f.close()

    # Check False Rejection Rate
    os.system(command.format(h=h, alpha=alpha, dataset=anomalous_dataset))
    with open ("loss.txt", 'r') as f:
        FRR = 1 - float(f.readline())
        f.close()
    
    return FAR + FRR



'''
Run the Black-Box Optimizer
'''
model             = "pca"
optimizer_name    = "NGOpt"
nominal_dataset   = "./datasets/gaussian_0_1.csv" # safe samples
anomalous_dataset = "./datasets/gaussian_1_1.csv" # anomalous samples

num_workers       = 4
num_iterations    = 250 * num_workers # write it as budget-per-worker

#      h     alpha
lb = [ 1.0,  0.01] # lower-bound
ub = [10.0,  1.00] # upper bound
instrumentation = ng.p.Instrumentation(
	h = ng.p.Log(lower=lb[0], upper=ub[0]),
	alpha = ng.p.Log(lower=lb[1], upper=ub[1])
)


optimizer = ng.optimizers.registry[optimizer_name](instrumentation, budget=num_iterations, num_workers=num_workers)

# do the job
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

# visualize result
print(recommendation.kwargs)
os._exit(0)