import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

from models.GNN import Network


if __name__ == '__main__':
    network = Network(train=False)
    network.load_simulation('./records/3d_nodes_simulation_101')

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    G = nx.DiGraph()
    colormap = []

    nodePos = []
    nodeIdcForGraph = []

    for inputIdx in network.inputIdc[:len(network.inputIdc) // 1]:
        G.add_node(inputIdx)
        nodeIdcForGraph.append(inputIdx)
        nodePos.append(network.P[inputIdx])
        colormap.append('red')

    for hiddenIdx in network.hiddenIdc[:len(network.hiddenIdc) // 1]:
        G.add_node(hiddenIdx)
        nodeIdcForGraph.append(hiddenIdx)
        nodePos.append(network.P[hiddenIdx])
        colormap.append('green')

    for outputIdx in network.outputIdc[:len(network.outputIdc) // 1]:
        G.add_node(outputIdx)
        nodeIdcForGraph.append(outputIdx)
        nodePos.append(network.P[outputIdx])
        colormap.append('blue')

    edges = []
    edgesPerNode = np.zeros((network.N,))
    numEdgesPerNode = 1

    for i in nodeIdcForGraph:
        for j in nodeIdcForGraph:
            if network.W[i, j] != 0 \
                    and edgesPerNode[i] < numEdgesPerNode:
                edges.append((i, j))
                edgesPerNode[i] += 1

    for edge in edges:
        G.add_edge(*edge)

    nodePos = np.array(nodePos)
    edgePos = np.array([
        (nodePos[nodeIdcForGraph.index(u)],
         nodePos[nodeIdcForGraph.index(v)]) for u, v in G.edges
    ])

    ax.scatter(*nodePos.T, s=100, c=colormap)
    # for i, pos in enumerate(nodePos):
    #     ax.text(*(pos - 1), nodeIdcForGraph[i], fontsize=8)

    for vizedge in edgePos:
        ax.plot(*vizedge.T, color="tab:gray")

    def _format_axes(ax):
        """Visualization options for the 3D axes."""
        # Turn gridlines off
        ax.grid(False)
        # Suppress tick labels
        for dim in (ax.xaxis, ax.yaxis, ax.zaxis):
            dim.set_ticks([])
        # Set axes labels
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

    _format_axes(ax)
    fig.tight_layout()
    plt.show()
