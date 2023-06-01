import os


def get_foldername():
    simulationIdx = 0
    while True:
        filename = f'3d_nodes_simulation_{simulationIdx}'

        if os.path.exists(f'./records/{filename}'):
            simulationIdx += 1
        else:
            break

    return filename
