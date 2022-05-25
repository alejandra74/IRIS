# Calculate mean distance between points
def mean_distance(matrix_x, matrix):
    # import packages
    from itertools import permutations
    from scipy.spatial import distance as dist
    import numpy as np

    # create landmark distance matrix
    d_landmk = [[] for _ in range(len(matrix_x))]
    # create array for mean distance
    media = []
    for i in range(len(matrix_x)):
        if len(matrix[i]) != 0:
            permut = permutations(matrix[i], 2)
            b = list(permut)
            n = len(matrix[i])
            if n != 1:
                for k in range(n * (n - 1)):
                    d_landmk[i].append(dist.euclidean(b[k][0], b[k][1]))
                arr = np.asarray(d_landmk[i]).reshape(n, n - 1)
                iu = np.triu_indices(n - 1)
                arr2 = []
                for j in arr[iu]:
                    if j > 5:
                        arr2.append(j)
                if not arr2 == []:
                    a = np.mean(arr2)
                    media.append(a)
                else:
                    media.append(0)
        else:
            d_landmk[i].append(0)
            media.append(0)

    while len(media) != len(matrix_x):
        media.append(0)

    return media
