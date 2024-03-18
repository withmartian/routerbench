import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def calculate_area_under_curve(points: np.ndarray):
    assert points.shape[1] == 2
    return np.trapz(points[:, 1], points[:, 0])


def get_non_descreasing_convex_hull_of(points: np.ndarray):
    """
    :param points: a numpy array of shape (n, 2)
    :return: a numpy array of shape (m, 2), where m <= n
    """
    convex_hull_points = points
    # Sort by x value
    convex_hull_points = convex_hull_points[convex_hull_points[:, 0].argsort()]

    # Go through all pairs of points
    new_convex_hull_points = []
    for i in range(len(convex_hull_points)):
        for j in range(i, len(convex_hull_points)):
            x1, y1 = convex_hull_points[i]
            x2, y2 = convex_hull_points[j]
            for k in range(len(convex_hull_points)):
                x, y = convex_hull_points[k]
                if x1 < x < x2:
                    # Interpolate between the two points
                    interpolated_y_value = np.interp(x, [x1, x2], [y1, y2])
                    if y < interpolated_y_value:
                        if x not in [mx for mx, _ in new_convex_hull_points]:
                            new_convex_hull_points.append([x, interpolated_y_value])
                        else:
                            for l in range(len(new_convex_hull_points)):
                                x_new, y_new = new_convex_hull_points[l]
                                if x_new == x and y_new < interpolated_y_value:
                                    new_convex_hull_points[l] = [
                                        x,
                                        interpolated_y_value,
                                    ]
                                    break
                                elif x_new == x and y_new > interpolated_y_value:
                                    new_convex_hull_points[l] = [x, y_new]
                    if y >= interpolated_y_value:
                        if x not in [mx for mx, _ in new_convex_hull_points]:
                            new_convex_hull_points.append([x, y])
                        else:
                            for l in range(len(new_convex_hull_points)):
                                x_new, y_new = new_convex_hull_points[l]
                                if x_new == x and y_new < y:
                                    new_convex_hull_points[l] = [x, y]
                                    break
                                elif x_new == x and y_new > y:
                                    new_convex_hull_points[l] = [x, y_new]
    # If the last point is not in the new_convex_hull_points, and the y value is larger than the last point, add it
    if convex_hull_points[-1][0] not in [mx for mx, _ in new_convex_hull_points]:
        if convex_hull_points[-1][1] >= new_convex_hull_points[-1][1]:
            new_convex_hull_points.append(convex_hull_points[-1])
    # If the first point is not in the new_convex_hull_points, then add it to the beginning
    if convex_hull_points[0][0] not in [mx for mx, _ in new_convex_hull_points]:
        new_convex_hull_points = [convex_hull_points[0]] + new_convex_hull_points
    new_convex_hull_points = np.array(new_convex_hull_points)
    convex_hull_points = new_convex_hull_points
    convex_hull_points = convex_hull_points[convex_hull_points[:, 0].argsort()]
    # Remove any points which have a smaller y value than the previous point
    points_to_remove = []
    for i in range(1, len(convex_hull_points)):
        if convex_hull_points[i][1] < convex_hull_points[i - 1][1]:
            points_to_remove.append(i)
    convex_hull_points = np.delete(convex_hull_points, points_to_remove, axis=0)
    return convex_hull_points


def calc_AIQ(
    router_results_list: list,
    plot_graph: bool = True,
    graph_title_eval_name: str = None,
    router_results_names: list = None,
):
    """
    input a list of router_results, return the AIQ for each of them in a list in order
    note that each of the dataframe in the list should have the same eval_name

    process:
    1. for each of the router_result, find the convex hull
    2. find the largest co
    """
    for router_result in router_results_list:
        assert (
            router_result["eval_name"].nunique() == 1
            and router_result["eval_name"].iloc[0] == eval_name
        )

    # find the list of (non-decreasing) convex hulls for each of the router_result
    ndch_list = []
    for router_result in router_results_list:
        points = router_result[["total_cost", "performance"]].values
        ndch_points_sorted = get_non_descreasing_convex_hull_of(points)
        ndch_list.append(ndch_points_sorted)

    # among all the points, find the one with largest x value
    max_x_value = max([np.max(ndch[:, 0]) for ndch in ndch_list])
    # for each array in ndch_list, add a point (max_x_value, last_y_value_in_that_ndch_list) to the end
    ndch_list = [
        np.vstack((ndch, np.array([max_x_value, ndch[-1, 1]]))) for ndch in ndch_list
    ]
    # calculate the area under curve for each of the ndch_list, normalize the x range from 0 to 1
    AIQs = [calculate_area_under_curve(ndch) / max_x_value for ndch in ndch_list]

    if plot_graph:
        for i in range(len(ndch_list)):
            ndch = ndch_list[i]
            plt.plot(
                ndch[:, 0],
                ndch[:, 1],
                label=router_results_names[i],
                linestyle="--",
                marker="x",
            )
            # label the area under curve of this ndch
            plt.text(ndch[-1, 0], ndch[-1, 1], f"AIQ: {AIQs[i]}")
        plt.legend()
        plt.title(f"AIQ for {graph_title_eval_name}")
        plt.show()
        plt.close()

    return AIQs


if __name__ == "__main__":
    router_results = pd.read_csv("routerbench_results.csv")
    overall_AIQs = {}
    overall_AIQs_names = {}
    for eval_name in router_results["eval_name"].unique():
        router_name = "knn"

        router_res_1 = router_results[router_results["eval_name"] == eval_name]
        router_res_1 = router_res_1[router_res_1["model_name"] == router_name]
        router_dfs_to_calc_AIQ, router_dfs_names = [], []
        for embedding in router_res_1.embedding.unique():
            embedding_df = router_res_1[router_res_1["embedding"] == embedding]
            # Only add it if it has more than 1 unique performance, cost pair
            if len(embedding_df[["total_cost", "performance"]].drop_duplicates()) > 2:
                router_dfs_to_calc_AIQ.append(embedding_df)
                router_dfs_names.append(f"embedding: - {embedding}")

        AIQs = calc_AIQ(
            router_dfs_to_calc_AIQ,
            plot_graph=True,
            router_results_names=router_dfs_names,
            graph_title_eval_name=eval_name,
        )
        # print the AIQ and the corresponding error rate
        print(eval_name)
        # Print the AIQ in the order from highest to lowest, with the correct name
        AIQs, router_dfs_names = zip(*sorted(zip(AIQs, router_dfs_names), reverse=True))
        overall_AIQs[eval_name] = AIQs
        overall_AIQs_names[eval_name] = router_dfs_names
        for i in range(len(AIQs)):
            print(f"{router_dfs_names[i]}: {AIQs[i]}")
    average_AIQ_per_router = {name: [] for name in overall_AIQs_names["mmlu"]}
    for eval_name in overall_AIQs.keys():
        for i in range(len(overall_AIQs[eval_name])):
            average_AIQ_per_router[overall_AIQs_names[eval_name][i]].append(
                overall_AIQs[eval_name][i]
            )
    print("Overall Average AIQ")
    # Print in the order from highest to lowest
    average_AIQ_per_router = {
        name: np.mean(value) for name, value in average_AIQ_per_router.items()
    }
    average_AIQ_per_router = dict(
        sorted(average_AIQ_per_router.items(), key=lambda item: item[1], reverse=True)
    )
    for name, value in average_AIQ_per_router.items():
        print(f"{name}: {np.mean(value)}")
