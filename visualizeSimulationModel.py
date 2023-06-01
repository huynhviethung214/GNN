import os
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import json


if __name__ == '__main__':
    simulationFolderPath = './records/3d_nodes_simulation_37'

    try:
        os.mkdir(f'{simulationFolderPath}/model_graph/')
    except FileExistsError as e:
        pass

    with open(f'{simulationFolderPath}/configs.json', 'r') as jsf:
        config = json.load(jsf)

    N = config['N']
    nInputs = config['nInputs']
    nOutputs = config['nOutputs']

    inputLayerPerFragment = 1
    numberOfInputNodePerLayerPerFragment = 1
    fragmentSize = numberOfInputNodePerLayerPerFragment * inputLayerPerFragment
    numberOfFragments = nInputs // fragmentSize

    print(f'Total number of fragments: {numberOfFragments}')

    inputIdc = [i for i in range(nInputs)]  # NOQA
    outputIdc = [N - i for i in range(1, nOutputs + 1)]  # NOQA
    hiddenIdc = []

    copiedInputIdc = inputIdc.copy()
    copiedInputIdc.extend(outputIdc)

    for i in range(N):  # NOQA
        if i not in copiedInputIdc:
            hiddenIdc.append(i)

    W = np.load(f'{simulationFolderPath}/w.npy')

    print(f'InputIdc  -> Start Idx: {inputIdc[0]}, End Idx: {inputIdc[-1]}')
    print(f'HiddenIdc -> Start Idx: {hiddenIdc[0]}, End Idx: {hiddenIdc[-1]}')
    print(f'OutputIdc -> Start Idx: {outputIdc[0]}, End Idx: {outputIdc[-1]}')

    for fragmentIdx in range(numberOfFragments):
        # Visualize Model
        G = nx.DiGraph()

        # Coloring Nodes (input = 'red', hidden = 'green', output = 'blue') & Add Node
        colorMap = []
        lc = 0
        totalConnectionsPerNode = 0

        nodes = []
        for i, inputIdx in enumerate(inputIdc[fragmentIdx * fragmentSize: fragmentIdx * fragmentSize + fragmentSize]):
            nodesIdx = np.where(W[inputIdx] != 0)[0]
            nodes.append(inputIdx)

            if len(nodesIdx) >= 10:
                colorMap.append('r')
            elif len(nodesIdx) < 10:
                colorMap.append('o')
            else:
                colorMap.append('black')

            totalConnectionsPerNode += len(nodesIdx)
            G.add_node(inputIdx,
                       size=4,
                       layer=lc)

            if (i + 1) % numberOfInputNodePerLayerPerFragment == 0:
                lc += 1

        for hiddenIdx in hiddenIdc:
            nodes.append(hiddenIdx)
            colorMap.append('g')
            G.add_node(hiddenIdx,
                       size=4,
                       layer=lc)
        lc += 1

        for outputIdx in outputIdc:
            colorMap.append('b')
            G.add_node(outputIdx,
                       size=4,
                       layer=lc)

        # Add Edge
        for nodeIdx in nodes:
            connectedNodes = np.where(W[nodeIdx] != 0)[0]

            for connectedNode in connectedNodes:
                G.add_edge(nodeIdx, connectedNode)

        print(f'\nNumber of nodes in fragment {fragmentIdx}: {len(nodes)} (I: {numberOfInputNodePerLayerPerFragment}, H: {len(hiddenIdc)}, O: {nOutputs})')
        print(f'Number of trainable parameters: {len(np.where(W != 0)[0])}')
        print(f'Average connections / node in fragment {fragmentIdx}: {totalConnectionsPerNode // fragmentSize}')

        fig = plt.figure(3, figsize=(30, 30))
        pos = nx.multipartite_layout(G, subset_key="layer")
        nx.draw(G,
                with_labels=True,
                font_weight='bold',
                node_color=colorMap,
                pos=pos,
                font_size=5)
        plt.savefig(f'{simulationFolderPath}/model_graph/{fragmentIdx}.pdf')
        fig.clear()
