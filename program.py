
smallest_arr = None


def crush(li, uniques=None, unique_indices=None):
    global smallest_arr

    uniques_was_none = False
    if uniques is None:
        uniques_was_none = True

        uniques = []
        unique_indices = []
        last_value = li[0]
        starting_index = 0

        for index, value in enumerate(li):
            if value != last_value:
                if index - starting_index >= 3:
                    uniques.append(last_value)
                    unique_indices.append((starting_index, index))

                starting_index = index

            last_value = value

            # print(uniques)
            # print(unique_indices)

    for i, unique in enumerate(uniques):
        unique_starting_index, unique_ending_index = unique_indices[i]

        if uniques_was_none and unique == li[-1]:
            smallest_arr = li

        new_li = li[:unique_starting_index] + li[unique_ending_index+1:]
        new_uniques = uniques[:i] + uniques[i+1:]
        new_unique_indices = unique_indices[:i] + unique_indices[i+1:]

        print(unique, new_li, new_uniques, new_unique_indices)

        crush(new_li, new_uniques, new_unique_indices)


def main():
    li = [1, 3, 3, 3, 2, 2, 2, 3, 1]

    crush(li)
    print(smallest_arr)


if __name__ == "__main__":
    main()
