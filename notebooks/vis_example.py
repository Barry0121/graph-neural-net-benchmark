"""
Example to create a spring layout plot based on closeness centrality
"""

import networkx as nx
import matplotlib.pyplot as plt

# for the example, I used the twitch data to populate a graph
twitch_edges = pd.read_csv("../data/raw/snap/twitch/ENGB/musae_ENGB_edges.csv") # get edges, skip 1 to skip header
twitch_target = pd.read_csv("../data/raw/snap/twitch/ENGB/musae_ENGB_target.csv") # get target
twitch_target = twitch_target.set_index('new_id')

twitch = nx.Graph()
twitch_nodes = twitch_target.index.unique().to_numpy()
twitch.add_nodes_from(twitch_nodes)
# twitch_edges = twitch_edges.to_numpy(dtype='int')
# twitch.add_edges_from(twitch_edges) # either one works

# create graph visualization (centrality plot)
pos = nx.spring_layout(twitch)
close_cent = nx.closeness_centrality(twitch)

# filter the nodes with more than 10000 viewers
node_views = {node: twitch.nodes[node]['views'] for node in twitch.nodes}

# I filtered out some nodes, but this isn't necessary
nodes_to_plot = [node for node, views in node_views.items() if views >= 10000]
edges_to_plot = [(node1, node2) for node1 in nodes_to_plot for node2 in nodes_to_plot if node1 != node2 and twitch.has_edge(node1, node2)]

# create graph visualization part 2
node_color = [10000.0 * close_cent[v] for v in nodes_to_plot]
node_size =  [close_cent[node]**2 * 20 for node in nodes_to_plot]

# plot the network graph
figure = plt.figure(figsize=(10,10))
nx.draw_networkx(twitch,
                    pos=pos,
                    with_labels=False,
                    node_color=node_color,
                    node_size=node_size,
                    nodelist=nodes_to_plot,
                    edgelist=edges_to_plot,
                    width=0.1)
figure.set_facecolor('white')
plt.axis('off')