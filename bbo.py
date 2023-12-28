import nevergrad as ng
import os

'''
This function performs the offline phase of the PCA model
'''
def pca_offline():
    command = "./anomaly_detector n 1.0 1.0 y y ./datasets/nominal-human-activity.csv"
    os.system(command)
    print("PCA offline phase done.")

'''
This function will call the PCA model and returns the loss value
'''
def pca_online(h:float, alpha:float) -> float:
    # execute model
    nominal_dataset = "./datasets/nominal-human-activity.csv" # non-anomalous samples
    anomalous_dataset = "./datasets/nominal-human-activity.csv" # anomalous samples
    command = "./anomaly_detector n {h} {alpha} n y {dataset}"

    # Check False Acceptance Rate
    os.system(command.format(h=h, alpha=alpha, dataset=nominal_dataset))
    f = open("loss.txt", "r")
    FAR = float(f.readline())
    f.close()

    # Check False Negative Rate
    # os.system(command.format(h=h, alpha=alpha, dataset=anomalous_dataset))
    # f = open("loss.txt", "r")
    # FNR = 1 - float(f.readline())
    # f.close()

    # print("Current loss: ", str(loss))
    return FAR #+ FNR


'''
This function performs the offline phase of the GEM model
'''
def gem_offline():
    command = "./anomaly_detector y 1.0 1.0 y y ./datasets/nominal-human-activity.csv"
    os.system(command)
    print("PCA offline phase done.")

'''
This function will call the PCA model and returns the loss value
'''
def gem_online(h:float, alpha:float) -> float:
    # execute model
    nominal_dataset = "./datasets/nominal-human-activity.csv" # non-anomalous samples
    anomalous_dataset = "./datasets/nominal-human-activity.csv" # anomalous samples
    command = "./anomaly_detector y {h} {alpha} n y {dataset}"

    # Check False Acceptance Rate
    os.system(command.format(h=h, alpha=alpha, dataset=nominal_dataset))
    f = open("loss.txt", "r")
    FAR = float(f.readline())
    f.close()

    # Check False Negative Rate
    # os.system(command.format(h=h, alpha=alpha, dataset=anomalous_dataset))
    # f = open("loss.txt", "r")
    # FNR = 1 - float(f.readline())
    # f.close()

    # print("Current loss: ", str(loss))
    return FAR #+ FNR



'''
Run the Black-Box Optimizer
'''
optimizer_name = "NGOpt"
#      h     alpha
lb = [ 1.0,  0.01] # lower-bound
ub = [10.0,  1.00] # upper bound

instrumentation = ng.p.Instrumentation(
	h = ng.p.Log(lower=lb[0], upper=ub[0]),
	alpha = ng.p.Log(lower=lb[1], upper=ub[1])
)
num_workers = 1
num_iterations = 5 * num_workers # budget per worker

# Let us create a Nevergrad optimization method.
optimizer = ng.optimizers.registry[optimizer_name](instrumentation, budget=num_iterations, num_workers=num_workers)

# do the job
#pca_offline() # only do this the first time!!
recommendation = optimizer.minimize(pca_online, verbosity=2)

# visualize result
print(recommendation.kwargs)
os._exit(0)