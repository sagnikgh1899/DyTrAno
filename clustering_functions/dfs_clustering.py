# TODO: Improve the variable and function names

# """
# DFS Based Approach:
# Due to the maximum recursion depth constraint in Python Programming language
# this is not a very good approach and will fail for large datasets.
# The BFS one is a better solution.
# """

# def NovelClus(Parent, Hash_Map, idx, dataset, K, list_of_parents, labels, thresh, cluster_num, can_form_cluster):
#     #print(Parent[1], idx)

#     print("Cluster =", cluster_num)

#     while idx <= K-2:

#         Density_Parent = Parent[0]
#         Radius = Parent[2]
#         Child_Idx = Parent[3][idx]
#         Child = Hash_Map[Child_Idx]
#         Child_Datapoint = Child[1]

#         # if child is already labelled then return
#         if labels[Child_Idx] > 0:
#             idx += 1
#             continue


#         neigh = NearestNeighbors(radius = Radius)
#         neigh.fit(dataset)
#         rng = neigh.radius_neighbors([Child_Datapoint])
#         #print(rng[0])

#         Density_Child = sum(rng[0][0])/len(rng[0][0])

#         # if dis-density threshold criteria is satisfied
#         if Density_Parent not in list_of_parents:
#             list_of_parents.append(Density_Parent)
#         data_parent = np.array(list_of_parents, dtype=np.float64)
#         weighted_density = numpy_ewma_vectorized_v2(data_parent, alpha_weight_avg)
#         print("Density of Parent =", weighted_density)
#         print("Density of Child = ", Density_Child)
#         numerator = abs(weighted_density - Density_Child)
#         denominator = weighted_density
#         print("Weighted Density =", numerator/denominator)

#         if numerator/denominator <= thresh:
#             labels[Child_Idx] = cluster_num
#             can_form_cluster[0] = [True]
#             NovelClus(Child, Hash_Map, 0, dataset, K, list_of_parents, labels, thresh, cluster_num, can_form_cluster)

#         # else term Child as "Anomaly" and return
#         else:
#             #print("\n")
#             labels[Child_Idx] = -1
#             idx += 1
#             continue
