def matrix_generator(x,y,e,d):
    # create matrix that save coord points of countor that match each body part
    matrix_x = [[] for _ in range(len(x))]
    matrix_y = [[] for _ in range(len(y))]
    for i in range(len(y)):
        for j in range(len(e)):
            if y[i] == e[j] or y[i] + 1 == e[j] or y[i] - 1 == e[j]:
                matrix_y[i].append(e[j])
                matrix_x[i].append(d[j])
    # create matrix with elements from matrix_x and matrix_y
    matrix = [[] for _ in range(len(matrix_x))]
    for i in range(len(matrix_x)):
        for j in range(len(matrix_x[i])):
            matrix[i].append((matrix_x[i][j], matrix_y[i][j]))

    return matrix, matrix_x, matrix_y