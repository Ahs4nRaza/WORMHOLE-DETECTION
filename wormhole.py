import numpy as np
import math
import matplotlib.pyplot as plt
import networkx as nx
import scipy.spatial
import random


# function to generate 2D WSN graph

def generate_graph(num_nodes=400,radius=1.2,side_len=10,noise=1):
    assert(math.sqrt(num_nodes).is_integer()),'square root must be int' #check for condition that num_nodes is square or not
    # genrating random graphs
    for X in range(0,1000):
        h=int(math.sqrt(num_nodes))
        node_dist=side_len/h
        max_noise=node_dist*noise
        con_graph=nx.Graph() 
        con_graph.add_nodes_from(range(0,num_nodes)) # added new sensor node
        positions_=list() # a variable to store node's positions
        positions=dict()
        for i in range(0, h):
            for j in range(0, h):
                #introducing randomness
                rx = random.uniform(-1*max_noise, max_noise)
                ry = random.uniform(-1*max_noise, max_noise)
                # adding nodes at newly calculated x and y coordinates
                positions[i + j * h] = [i * node_dist + rx, j * node_dist + ry]
                positions_.append(positions[i + j * h])
        y = scipy.spatial.distance.pdist(np.asarray(positions_), 'euclidean')
        dist_matrix = scipy.spatial.distance.squareform(y)
        for i in range(0, num_nodes-1):
            for j in range(i, num_nodes):
                if dist_matrix[i, j] <= radius and i != j:
                    con_graph.add_edge(i, j)

        if nx.is_connected(con_graph):
            return positions, con_graph
    # if wrong parameters entered so exit without forming graphs   
    assert False, 'Parameters cant create graph\n'    

# selecting some nodes as wormholes nodes
def add_wormhole(positions, con_graph, worm_radius=.6, min_dist=6):
    total = len(positions)
    for X in range(0,1000):
        new_graph=con_graph.copy() #creating a copy in order to prevent unexpected changes in orginal graph
        hole1=np.random.randint(0,total) #select a random node as the first wormhole.
        distances = nx.shortest_path_length(con_graph, source=hole1)
        candidate_hole2 = [node for node in distances if distances[node] > min_dist] # select a node as second wormhole if satifies conditions
        if len(candidate_hole2) == 0:
            continue
        hole2=random.choice(candidate_hole2)
        Y = scipy.spatial.distance.pdist(np.asarray(list(positions.values())), 'euclidean')
        dist_matrix = scipy.spatial.distance.squareform(Y)
        #create a list of nodes affected by the wormholes, starting with the first worm
        affected1 = [hole1]
        affected2 = [hole2]
        for p in range(0,total):
            if p!= hole2 and dist_matrix[hole2,p]< worm_radius:
                affected2.append(p)
        for p in range(0,total):
            if p!= hole1 and dist_matrix[hole1,p]< worm_radius:
                affected1.append(p)
        # create new short links changing the transmission links
        for wormhole_point1 in affected1:
            for wormhole_point2 in affected2:
                new_graph.add_edge(wormhole_point1, wormhole_point2)

        return new_graph, affected1, affected2
    assert len(candidate_hole2) > 0, 'Parameters cant create graph\n'
    
def get_root_nodes(new_graph, min_dist=7):
    
    nodes = [i for i in range(0, len(new_graph))]
    root_nodes = [] # to store selected pivot points

    while len(nodes) != 0:
        root_node = random.choice(nodes) # select a random node as root
        root_nodes.append(root_node)
        distances = nx.shortest_path_length(new_graph, source=root_node) # get immediate neighbours
        to_rem = [key for key in distances if distances[key] < min_dist] # remove neighbours
        nodes = [x for x in nodes if x not in to_rem]

    assert len(root_nodes) > 1,'Parameters cant create graph\n' # check to see if we have more one root node as required

    return root_nodes

def find_wormhole(new_graph, root_nodes, affected, threshold=5):
    #Create a 2D array to store the variance values for each root node and each node in the graph
    plt.figure(1, figsize=(5, 10))
    var_matrix = np.zeros(shape=(len(root_nodes), len(new_graph)))  
    
      # Loop over each root node and run the detection algorithm
    for idx, root_node in enumerate(root_nodes):
        print('running detection from root {}/{}'.format(idx+1, len(root_nodes)))
        plt.subplot(len(root_nodes)+1, 1, (idx+1))
        plt.title('root {}:'.format(idx), loc='left')
        
        # Create a list of nodes to bypass (including the root node and its neighbors)
        to_bypass = [root_node] + list(new_graph.neighbors(root_node))
        original_distances = nx.shortest_path_length(new_graph, root_node) # find distances of each node from root

        for node in range(len(new_graph)):  # Loop over each node in the graph
            if node not in to_bypass:
                tmp_graph = new_graph.copy()
                nodes_to_delete = [node] + list(new_graph.neighbors(node))
                # if node is not in bypass list -->exclude itself and its  neighbours and make new graph
                for node_to_delete in nodes_to_delete:
                    tmp_graph.remove_node(node_to_delete)
                   # find new distances of the remaining graph 
                new_distances = nx.shortest_path_length(tmp_graph, root_node) 
                #find the diff of distances between orignal and new graph
                distance_differences = [abs(original_distances[key] - new_distances[key]) for key in new_distances]
                variance_in_dd = np.var(distance_differences)
                var_matrix[idx, node] = variance_in_dd # find variance 
                plt.plot(node, variance_in_dd, color='#00ff00', marker='.')
                if node in affected:
                    plt.plot(node, variance_in_dd, 'bo')
            if node in to_bypass: #not to be checked
                var_matrix[idx, node] = np.nan
    avgs = list()
    for x in range(var_matrix.shape[1]):
        to_avg = var_matrix[:, x]
        avg = np.nanmean(to_avg)
        avgs.append(avg)

    final_threshold = np.mean(np.asarray(avgs))*threshold # threshold value 
    candidates = [x for x in range(len(avgs)) if avgs[x] > final_threshold] # check for wormhole --> if true add to candidate list
    inducted_graph = new_graph.subgraph(candidates)
    for cc in nx.connected_components(inducted_graph):
        if len(cc) == 1:
            candidates.remove(list(cc)[0])
        plt.subplot(len(root_nodes)+1, 1, len(root_nodes)+1)
        for node in range(0, len(new_graph)):
            plt.plot(node, avgs[node], color='#00ff00', marker='.')
            if node in candidates:
                plt.plot(node, avgs[node], color='#ff99ff', marker='D', markersize=6)
            if node in affected:
                plt.plot(node, avgs[node], color='b', marker='o', markersize=3)
        plt.plot([0, len(new_graph)], [final_threshold, final_threshold], 'r-')

    return candidates


def main():
    print('Generating sensor network...')
    positions, con_graph=generate_graph()
    print('Done.')

    degrees = nx.degree(con_graph)
    avg_degree = np.mean([deg for node, deg in degrees])
    max_degree = np.max([deg for node, deg in degrees])

    print('\navg. # of neighbours', avg_degree)
    print('max. # of neighbours', max_degree)
    
    print('Inserting wormhole...')
    new_graph, affected1, affected2 = add_wormhole(positions, con_graph)
    print('Done.')

    
    print('Choosing root points...')
    root_nodes = get_root_nodes(new_graph)
    print('Done.')

    print('\nRunning wormhole detection...')
    candidates = find_wormhole(new_graph, root_nodes,affected1+affected2)
    print('Done.')

    positives = set(candidates)
    negatives = set([x for x in range(0, len(new_graph))]).difference(positives)
    wormholes = set(affected1+affected2)
    not_wormholes = set([x for x in range(0, len(new_graph))]).difference(wormholes)

    true_positives = positives.intersection(wormholes)
    false_positives = positives.intersection(not_wormholes)
    

    print ('\nResults :')
    print ("False Positive Count : " , len(false_positives))
    print ("True Positive Count : " , len(true_positives))
    
    
    plt.figure(2)
    nx.draw_networkx(con_graph, pos=positions, node_size=10, linewidths=0, edge_color='#00ff00', with_labels=False)

    for node in candidates:  # candidates = pink
        plt.plot(positions[node][0], positions[node][1], color='#ff99ff', marker='D', markersize=4)

    for endpoint in (affected2+affected1):  # wormhole node = blue
        plt.plot(positions[endpoint][0], positions[endpoint][1],  color='b', marker='o', markersize=2)

    for idx, root_node in enumerate(root_nodes):  # root nodes = black
        plt.plot(positions[root_node][0], positions[root_node][1], marker='D', color='k', markersize=4)
        plt.annotate(str(idx), xy=(positions[root_node][0], positions[root_node][1]))

    plt.show()


main()
