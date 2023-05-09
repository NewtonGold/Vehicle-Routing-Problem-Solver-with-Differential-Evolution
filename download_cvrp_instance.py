import vrplib
import pickle

instances = ["A-n32-k5", "A-n33-k5", "A-n33-k6", "P-n16-k8", "P-n19-k2"]

# instance_name = "P-n16-k8"

for instance_name in instances:
    # Download an instance and a solution file 
    vrplib.download_instance(instance_name, "./cvrp_instances/" + instance_name + ".vrp")
    vrplib.download_solution(instance_name, "./cvrp_solutions/" + instance_name + ".sol")

    instance = vrplib.read_instance("./cvrp_instances/" + instance_name + ".vrp")
    global_solution = vrplib.read_solution("./cvrp_solutions/" + instance_name + ".sol")

    with open("./cvrp_instances/" + instance_name + ".pkl", 'wb') as f:
        pickle.dump(instance, f)

    with open("./cvrp_solutions/" + instance_name + ".pkl", 'wb') as f:
        pickle.dump(global_solution, f)



# List all instance names that can be downloaded 
# vrplib.list_names()                      # All instance names
# names = vrplib.list_names(low=5, high=100, vrp_type="cvrp", )     # Instances with between [100, 200] customers
# print(names)
# vrplib.list_names(vrp_type="cvrp")       # Only CVRP instances
# vrplib.list_names(vrp_type="vrptw")      # Only VRPTW instances