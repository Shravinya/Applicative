import time

def insertion_sort(arr):
    start_time = time.time()
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and key < arr[j]:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
    return arr, time.time() - start_time

def quick_sort(arr):
    start_time = time.time()
    
    def quicksort_helper(arr):
        if len(arr) <= 1:
            return arr
        pivot = arr[len(arr) // 2]
        left = [x for x in arr if x < pivot]
        middle = [x for x in arr if x == pivot]
        right = [x for x in arr if x > pivot]
        return quicksort_helper(left) + middle + quicksort_helper(right)
    
    sorted_arr = quicksort_helper(arr)
    return sorted_arr, time.time() - start_time

def merge_sort(arr):
    start_time = time.time()
    
    def merge_sort_helper(arr):
        if len(arr) <= 1:
            return arr
        mid = len(arr) // 2
        left = merge_sort_helper(arr[:mid])
        right = merge_sort_helper(arr[mid:])
        return merge(left, right)
    
    def merge(left, right):
        result = []
        i = j = 0
        while i < len(left) and j < len(right):
            if left[i] < right[j]:
                result.append(left[i])
                i += 1
            else:
                result.append(right[j])
                j += 1
        result.extend(left[i:])
        result.extend(right[j:])
        return result
    
    sorted_arr = merge_sort_helper(arr)
    return sorted_arr, time.time() - start_time
