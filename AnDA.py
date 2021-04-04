def predictions(catalogue, observations, method, k=100):
    """predict the next state of a list of tourbillons
    observations of shape Mx10, returns array of shape Mx10"""
    tourbillons_suivant=[]
    predecesseurs = catalogue[:,0,:]
    N = predecesseurs.shape[0]
    for tourbillon in observations:
        distances = new_wasserstein(np.stack([tourbillon]*N, axis=0), predecesseurs)
        indices_wt = np.argpartition(distances, k)[:k] # indices of the k nearest neighbors
        neighbors = catalogue[indices_wt,0,:]
        successors = catalogue[indices_wt,1,:]
        distances_neighbors = distances[indices_wt]
        weights = compute_weights(distances_neighbors, l=np.median(distances_neighbors))
        pred = method(tourbillon, neighbors, successors, weights)
        tourbillons_suivant.append(pred)
    return np.array(tourbillons_suivant)



def list_prediction(catalogue,nb_predictions, observations, method, k=50):
    """construit une matrice de taille nombre d'ellipses x nb_predictions x 10
    cette matrice représente les valeurs prédites"""
    mat_prediction = np.empty((observations.shape[0],nb_predictions,observations.shape[1]))
    next_obs = observations
    for j in range(nb_predictions):
        next_obs = predictions(catalogue,next_obs, method, k=k)
        mat_prediction[:,j] = next_obs
    return mat_prediction


def load_data(filename ='data\catalogue.txt'):
    data = np.loadtxt(filename)
    data = data.reshape((int(data.shape[0]/2),2,10))
    return data
