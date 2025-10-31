"""Simple bubble sort demonstration."""
from __future__ import annotations

from typing import List


def bubble_sort(values: List[int]) -> List[int]:
    """Return a sorted copy of the given list using bubble sort."""
    arr = values[:]
    n = len(arr)
    for i in range(n):
        swapped = False
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
        if not swapped:
            break
    return arr


def main() -> None:
    data = [64, 34, 25, 12, 22, 11, 90]
    print("原始数组:", data)
    sorted_data = bubble_sort(data)
    print("排序后:", sorted_data)


if __name__ == "__main__":
    main()
