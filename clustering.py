from heatmap import *
import matplotlib.pyplot as plt
from math import pi, cos, sin

#Computation of the center of the graph
def adjacency(data, path):
    n = len(data)
    adj = np.zeros((n,n))

    for i in range(n):
        for j in range(i,n):
            img1 = Image.open(path + "/" + data[i])
            img2 = Image.open(path + "/" + data[j])
            img1 = img1.convert('L')
            img2 = img2.convert('L')
            adj[i, j] = normImage(differenceImage(img1, img2))
            adj[j, i] = adj[i, j]

    return adj

def centralityScore(adj):
    """Returns the eigenvector centrality score for all nodes in a graph."""
    eigValues, eigVectors = np.linalg.eig(adj)
    i = np.argmax(eigValues)
    centralityScoreVector = eigVectors[i]

    return centralityScoreVector

def plotCircle(center, radius):
    T = np.linspace(0,2*pi)
    X = radius*np.cos(T) + center[0] ; Y = radius*np.sin(T) + center[1]
    plt.plot(X,Y)



if __name__ == "__main__":
    path = "Detection_Test_Set/Detection_Test_Set_Img"
    data = os.listdir(path)[:10]
    adj = adjacency(data, path)
    centerIndex = np.argmax(centralityScore(adj))


    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    for i in range(len(data)):
        plotCircle((0,0), adj[i, centerIndex])
    plt.show()



