# histogram.py
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from loader import getInputAndLabel
from params import selectedCathegory


def compute_homogeneity_score(values: np.ndarray, houses: np.ndarray, bins: int = 20):
    unique_houses = sorted(set(houses.tolist()))
    bin_edges = np.histogram(values, bins=bins)[1] # get the bin edge

    per_house_densities = []
    for house in unique_houses:
        house_values = values[houses == house] # value for this house
        density_counts, _ = np.histogram(house_values, bins=bin_edges, density=True) # % of student in the bin / size of the bin
        per_house_densities.append(density_counts)

    # Average pairwise L1 distance
    total_distance = 0.0
    pair_count = 0
    for i in range(len(per_house_densities)):
        for j in range(i + 1, len(per_house_densities)):
            total_distance += np.abs(per_house_densities[i] - per_house_densities[j]).mean()
            pair_count += 1

    score = total_distance / pair_count if pair_count > 0 else 0.0
    return score, bin_edges, unique_houses


def main():
    if len(sys.argv) != 2:
        print("Usage: python3 histogram.py <dataset.csv>")
        return

    # Load as arrays, then wrap features in a DataFrame for column-by-name access
    feature_matrix, house_labels = getInputAndLabel(sys.argv[1], selectedCathegory)
    houses_array = np.array(house_labels, dtype=str)
    df = pd.DataFrame(feature_matrix, columns=selectedCathegory)

    most_homogeneous_course_name = None
    most_homogeneous_score = None
    best_bin_edges = None
    best_unique_houses = None

    for course_name in selectedCathegory:
        course_values = df[course_name].to_numpy()
        score, bin_edges, unique_houses = compute_homogeneity_score(course_values, houses_array, bins=20)

        if (most_homogeneous_score is None) or (score < most_homogeneous_score):
            most_homogeneous_course_name = course_name
            most_homogeneous_score = score
            best_bin_edges = bin_edges
            best_unique_houses = unique_houses

    print(f"Most homogeneous course across houses: {most_homogeneous_course_name} (homogeneity score = {most_homogeneous_score:.6f})")

    # ---- Single histogram: per-house distributions for the best course ----
    plt.figure(figsize=(9, 6))
    best_values = df[most_homogeneous_course_name].to_numpy()

    for house in best_unique_houses:
        values_for_house = best_values[houses_array == house]
        plt.hist(
            values_for_house,
            bins=best_bin_edges,
            density=True,
            alpha=0.6,
            edgecolor="black",
            label=house,
        )

    plt.title(f"Distribution by House — {most_homogeneous_course_name}")
    plt.xlabel(most_homogeneous_course_name)
    plt.ylabel("Density")
    plt.legend(title="House")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
