import numpy as np

def fillArea(index, input, visited, threshold):
    area = [index]

    # We start with a single cell to check
    searchList = [index]

    while searchList != []:
        for d in range(len(searchList[0])):
            # In each dimension get the two adjacent cells by in- and decrementing the index
            # If a cell matches the criterion it is added to the list itself and in turn will have its neighbors checked
            leftCell = list(searchList[0])
            rightCell = list(searchList[0])
            leftCell[d] -= 1
            rightCell[d] += 1
            leftIndex = tuple(leftCell)
            rightIndex = tuple(rightCell)

            # Check the left dimensional neighbor for index out of bounds and the criterion
            if all(c >= 0 for c in leftIndex) and (visited[leftIndex] == 0) and input[leftIndex] >= threshold:
                searchList.append(leftIndex)
                visited[leftIndex] = 1

            # Check the right dimensional neighbor for index out of bounds and the criterion
            if all(rightIndex[i] < np.shape(input)[i] for i in xrange(len(rightIndex))) and visited[rightIndex] == 0 and input[rightIndex] >= threshold:
                searchList.append(rightIndex)
                visited[rightIndex] = 1

        # At last remove the cell from the search list and add it to the contiguous area
        area.append(searchList.pop(0))

    return area


def floodfill(input, threshold):
    areas = []
    # Keep track of the fields already visited (they cannot be part of multiple areas!)
    visited = np.zeros(np.shape(input), dtype=np.int)

    # Iterate over all fields
    for index, x in np.ndenumerate(input):
        # If it has not been visited yet (may happen in fillArea) and meets the criterion get the area it belongs to
        if visited[index] == 0 and input[index] >= threshold:
            visited[index] = 1
            areas.append(fillArea(index, input, visited, threshold))

    return areas


def sortByDistanceFromCenter(inds, varShape):
    # TODO: sort hypervolumes by their centre-of-mass-distance to the centre
    return inds