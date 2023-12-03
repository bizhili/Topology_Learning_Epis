import random
import logging
from datetime import datetime
import torch

def select_nodes_accroding_to_degree(G, strains, intense= 0):
    #G:
    #networkx grph object
    #strains:
    #select how much nodes
    #intense:
    #0: random select from these low degree nodes
    #1: random select from these mid degree nodes
    #2: random select from these high degree nodes
    sorted_nodes = sorted(G.nodes(), key=lambda x: G.degree(x))
    #print(sorted_nodes)
    #sorted_degree = [G.degree(i) for i in sorted_nodes]
    n= len(sorted_nodes)
    randomList= [random.randint(0, int(n/3))+int(n/3)*intense for _ in range(strains)]
    return [sorted_nodes[i] for i in randomList]



def get_time_string():
    # Get the current date and time
    current_time = datetime.now()

    # Format the time string
    time_string = current_time.strftime("%m_%d_%H_%M")

    return time_string

def log_print(flag, *args):
    if flag == 1:
        # If flag is 1, print to the screen
        print(*args)
    else:
        # If flag is not 1, log to a file
        logging.info(' '.join(map(str, args)))

def continious_to_sparcity(my_tensor, top= 400):
    # Flatten the array to a 1D array
    flat_tensor = my_tensor.flatten()

    # Get the indices of the top 400 elements
    top_indices = torch.topk(flat_tensor, k=int(top.item())).indices

    # Create a new tensor with zeros
    output_tensor = torch.zeros_like(flat_tensor)

    # Set the top 400 elements to 1
    output_tensor[top_indices] = 1

    # Reshape the tensor back to its original shape
    output_tensor = output_tensor.view(my_tensor.shape)

    return output_tensor
