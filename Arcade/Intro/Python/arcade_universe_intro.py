from hypothesis import given, strategies as st
import re
import numpy as np
from collections import Counter

"""Write a function that returns the sum of two numbers.

Example

For param1 = 1 and param2 = 2, the output should be
    solution(param1, param2) = 3."""


def add_two_nbrs(param1, param2):
    return param1 + param2


"""Given a year, return the century it is in. The first century spans 
from the year 1 up to and including the year 100, the second - from 
the year 101 up to and including the year 200, etc.

Example

For year = 1905, the output should be
    solution(year) = 20;
For year = 1700, the output should be
    solution(year) = 17."""


def find_century(year):
    if year < 100:
        return 1
    elif year % 100 == 0:
        return year // 100
    else:
        return year // 100 + 1


def century_from_year(year):
    # // -> Floor division - division that results into whole number adjusted to the left in the number line. Example: 15//4 = 3
    return 1 + (year - 1) // 100


"""Given the string, check if it is a palindrome.

Example

For inputString = "aabaa", the output should be
    solution(inputString) = true;
For inputString = "abac", the output should be
    solution(inputString) = false;
For inputString = "a", the output should be
    solution(inputString) = true."""


def is_palindrome(inputString):
    return inputString == inputString[::-1]


"""Given an array of integers, find the pair of adjacent elements that has the largest product and return that product.

Example

For inputArray = [3, 6, -2, -5, 7, 3], the output should be
    solution(inputArray) = 21.

7 and 3 produce the largest product."""


def adjacent_elements_product(inputArray):
    if (ln := len(inputArray)) <= 1:
        return 0

    largest_product = inputArray[0] * inputArray[1]

    for i in range(1, ln - 1):
        largest_product = max(inputArray[i] * inputArray[i + 1], largest_product)

    return largest_product


"""Below we will define an n-interesting polygon. Your task is to find the 
area of a polygon for a given n.

A 1-interesting polygon is just a square with a side of length 1. An 
n-interesting polygon is obtained by taking the n - 1-interesting polygon 
and appending 1-interesting polygons to its rim, side by side. You can 
see the 1-, 2-, 3- and 4-interesting polygons in the picture below.

https://app.codesignal.com/arcade/intro/level-2/

Example

For n = 2, the output should be
    solution(n) = 5;
For n = 3, the output should be
    solution(n) = 13. """


def shape_area(n):
    if n == 1:
        return 1

    area = 1
    for i in range(2, n + 1):
        area += (i - 1) * 4

    return area


"""Ratiorg got statues of different sizes as a present from CodeMaster for 
his birthday, each statue having an non-negative integer size. Since 
he likes to make things perfect, he wants to arrange them from smallest 
to largest so that each statue will be bigger than the previous one 
exactly by 1. He may need some additional statues to be able to accomplish 
that. Help him figure out the minimum number of additional statues needed.

Example

For statues = [6, 2, 3, 8], the output should be
    solution(statues) = 3.

Ratiorg needs statues of sizes 4, 5 and 7"""


def make_array_consecutive_1(statues):
    statues.sort()
    sm = 0
    for i in range(len(statues) - 1):
        sm += statues[i + 1] - statues[i] - 1

    return sm


def make_array_consecutive_2(statues):
    return max(statues) - min(statues) - len(statues) + 1


"""Given a sequence of integers as an array, determine whether it is possible 
to obtain a strictly increasing sequence by removing no more than one 
element from the array.

Note: sequence a0, a1, ..., an is considered to be a strictly increasing 
if a0 < a1 < ... < an. Sequence containing only one element is also 
considered to be strictly increasing.

Example

For sequence = [1, 3, 2, 1], the output should be
    solution(sequence) = false.

There is no one element in this array that can be removed in order to get 
a strictly increasing sequence.

For sequence = [1, 3, 2], the output should be
    solution(sequence) = true.

You can remove 3 from the array to get the strictly increasing sequence 
[1, 2]. Alternately, you can remove 2 to get the strictly increasing 
sequence [1, 3]."""


# my solution is ugly
def almost_increasing_sequenece(sequence):
    if (ln := len(sequence)) == 1:
        return True

    sequence_copy = sequence.copy()
    for i in range(ln - 1):
        if sequence[i] >= sequence[i + 1]:
            sequence.pop(i + 1)
            sequence_copy.pop(i)
            break

    return (
        len(set(sequence)) == len(sequence)
        and sequence == sorted(sequence)
        or len(set(sequence_copy)) == len(sequence_copy)
        and sequence_copy == sorted(sequence_copy)
    )


"""After becoming famous, the CodeBots decided to move into a new building 
together. Each of the rooms has a different cost, and some of them are free,
but there's a rumour that all the free rooms are haunted! Since the 
CodeBots are quite superstitious, they refuse to stay in any of the free 
rooms, or any of the rooms below any of the free rooms.

Given matrix, a rectangular matrix of integers, where each value represents 
the cost of the room, your task is to return the total sum of all rooms 
that are suitable for the CodeBots (ie: add up all the values that don't 
appear below a 0).

Example

For
matrix = [[0, 1, 1, 2], 
          [0, 5, 0, 0], 
          [2, 0, 3, 3]]
the output should be
    solution(matrix) = 9.

There are several haunted rooms, so we'll disregard them as well as any 
rooms beneath them. Thus, the answer is 1 + 5 + 1 + 2 = 9.

For
matrix = [[1, 1, 1, 0], 
          [0, 5, 0, 1], 
          [2, 1, 3, 10]]
the output should be
    solution(matrix) = 9.

Note that the free room in the final column makes the full column unsuitable 
for bots (not just the room directly beneath it). Thus, the answer is 1 + 
1 + 1 + 5 + 1 = 9."""


def matrix_elements_sum(matrix):
    sm = 0
    for i in range(len(matrix) - 1):
        for j in range(len(matrix[0])):
            if matrix[i][j] == 0:
                matrix[i + 1][j] = 0

    for row in matrix:
        sm += sum(row)

    return sm


def matrix_elements_sum_2(matrix):
    rows = len(matrix)
    cols = len(matrix[0])
    sm = 0
    for i in range(rows - 1):
        for j in range(cols):
            sm += matrix[i][j]
            if matrix[i][j] == 0:
                matrix[i + 1][j] = 0

    sm += sum(matrix[rows - 1])

    return sm


def matrix_elements_sum_3(matrix):
    arr = np.array(matrix)
    rows = len(arr)
    cols = len(arr[0])
    sm = 0
    for i in range(rows - 1):
        for j in range(cols):
            if arr[i][j] == 0:
                arr[i + 1][j] = 0

    return np.sum(arr)


"""Given an array of strings, return another array containing all of 
its longest strings.

Example

For inputArray = ["aba", "aa", "ad", "vcd", "aba"], the output should be
    solution(inputArray) = ["aba", "vcd", "aba"]."""


def all_longest_strings(inputArray):
    if len(inputArray) <= 1:
        return inputArray

    sortedlist = sorted(inputArray, key=lambda s: len(s), reverse=True)
    ln = len(sortedlist[0])

    return list([s for s in inputArray if len(s) == ln])


"""Given two strings, find the number of common characters between them.

Example

For s1 = "aabcc" and s2 = "adcaa", the output should be
    solution(s1, s2) = 3.

Strings have 3 common characters - 2 "a"s and 1 "c"."""

from collections import Counter


def common_charactoer_count(s1, s2):
    counter1 = Counter(s1)
    counter2 = Counter(s2)

    count = 0
    for k1, v1 in counter1.items():
        if k1 in counter2.keys():
            count += min(v1, counter2[k1])

    return count


def common_charactoer_count_2(s1, s2):
    counter1 = Counter(s1)
    counter2 = Counter(s2)

    count = 0
    char_set = set(s1)
    for c in char_set:
        if c in counter2.keys():
            count += min(counter1[c], counter2[c])

    return count


def common_charactoer_count_3(s1, s2):
    counter1 = Counter(s1)
    counter2 = Counter(s2)

    char_set1 = set(s1)
    char_set2 = set(s2)
    intersect = char_set1 & char_set2
    count = 0
    for c in intersect:
        count += min(counter1[c], counter2[c])

    return count


def common_charactoer_count_4(s1, s2):
    char_set1 = set(s1)
    char_set2 = set(s2)
    intersect = char_set1 & char_set2
    count = 0
    for c in intersect:
        count += min(s1.count(c), s2.count(c))

    return count


"""Ticket numbers usually consist of an even number of digits. A ticket number 
is considered lucky if the sum of the first half of the digits is equal to 
the sum of the second half.

Given a ticket number n, determine if it's lucky or not.

Example

For n = 1230, the output should be
    solution(n) = true;
For n = 239017, the output should be
    solution(n) = false."""

# number = 12345
# numList = [int(digit) for digit in str(number)]


def is_lucky(n):
    s = str(n)
    ln = len(s) // 2
    half1 = list(s[:ln])
    half2 = list(s[ln:])

    return sum([int(c) for c in half1]) == sum([int(c) for c in half2])


"""Some people are standing in a row in a park. There are trees between 
them which cannot be moved. Your task is to rearrange the people by their 
heights in a non-descending order without moving the trees. People can 
be very tall!

Example

For a = [-1, 150, 190, 170, -1, -1, 160, 180], the output should be
    solution(a) = [-1, 150, 160, 170, -1, -1, 180, 190]."""


def sort_by_height(a):
    a_copy = a.copy()
    a_copy.sort()
    a_copy = [n for n in a_copy if n != -1]
    k = 0
    for i in range(len(a)):
        if a[i] != -1:
            a[i] = a_copy[k]
            k += 1

    return a


"""Write a function that reverses characters in (possibly nested) parentheses 
in the input string.

Input strings will always be well-formed with matching ()s.

Example

For inputString = "(bar)", the output should be
    solution(inputString) = "rab";
For inputString = "foo(bar)baz", the output should be
    solution(inputString) = "foorabbaz";
For inputString = "foo(bar)baz(blim)", the output should be
    solution(inputString) = "foorabbazmilb";
For inputString = "foo(bar(baz))blim", the output should be
    solution(inputString) = "foobazrabblim".
Because "foo(bar(baz))blim" becomes "foo(barzab)blim" and then "foobazrabblim"."""


def reverse_parenthesis(inputString):
    if (n := inputString.count("(")) == 0:
        return inputString

    for i in range(n):
        start = inputString.rfind("(")
        end = inputString.find(")", start)
        if start == -1:
            break
        temp = inputString[start + 1 : end]
        inputString = inputString[:start] + temp[::-1] + inputString[end + 1 :]

    return inputString


"""Several people are standing in a row and need to be divided into two 
teams. The first person goes into team 1, the second goes into team 2, the 
third goes into team 1 again, the fourth into team 2, and so on.

You are given an array of positive integers - the weights of the people. 
Return an array of two integers, where the first element is the total 
weight of team 1, and the second element is the total weight of team 2 
after the division is complete.

Example

For a = [50, 60, 60, 45, 70], the output should be
solution(a) = [180, 105]."""


def alternating_sums_1(a):
    t1 = t2 = 0
    for k, v in enumerate(a):
        if k % 2 == 0:
            t1 += v
        else:
            t2 += v

    return [t1, t2]


def alternating_sums_2(a):
    return [
        sum([v for k, v in enumerate(a) if k % 2 == 0]),
        sum([v for k, v in enumerate(a) if k % 2 == 1]),
    ]


"""Given a rectangular matrix of characters, add a border of asterisks(*) to it.

Example

For

picture = ["abc",
           "ded"]
the output should be
    solution(picture) = ["*****",
                         "*abc*",
                         "*ded*",
                         "*****"]"""


# solution according to wrong description of list as matrix
def add_border(picture):
    rows = len(picture) + 2
    cols = len(picture[0]) + 2

    new_pic = [["*" for i in range(cols)] for j in range(rows)]
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            new_pic[i][j] = picture[i - 1][j - 1]

    return new_pic


def add_border_1(picture):
    ln = len(picture[0]) + 2
    top_bottome = "*" * ln
    lst = [top_bottome]
    picture = ["*" + s + "*" for s in picture]
    lst.extend(picture)
    lst.append(top_bottome)
    return lst


"""Two arrays are called similar if one can be obtained from another by 
swapping at most one pair of elements in one of the arrays.

Given two arrays a and b, check whether they are similar.

Example

For a = [1, 2, 3] and b = [1, 2, 3], the output should be
    solution(a, b) = true.

The arrays are equal, no need to swap any elements.

For a = [1, 2, 3] and b = [2, 1, 3], the output should be
    solution(a, b) = true.

We can obtain b from a by swapping 2 and 1 in b.

For a = [1, 2, 2] and b = [2, 1, 1], the output should be
    solution(a, b) = false.

Any swap of any two elements either in a or in b won't make a and b equal."""


def are_similar(a, b):
    if sum(a) != sum(b) or len(a) != len(b):
        return False

    if a == b:
        return True

    ln = len(a)
    for i in range(ln - 1):
        for j in range(i + 1, ln):
            a[i], a[j] = a[j], a[i]
            if a == b:
                return True
            a[j], a[i] = a[i], a[j]

    return False


"""You are given an array of integers. On each move you are allowed to 
increase exactly one of its element by one. Find the minimal number of 
moves required to obtain a strictly increasing sequence from the input.

Example

For inputArray = [1, 1, 1], the output should be
solution(inputArray) = 3."""


def array_change(inputArray):
    sm = 0
    for i in range(1, len(inputArray)):
        if inputArray[i] <= inputArray[i - 1]:
            sm += inputArray[i - 1] + 1 - inputArray[i]
            inputArray[i] = inputArray[i - 1] + 1

    return sm


"""Given a string, find out if its characters can be rearranged to form 
a palindrome.

Example

For inputString = "aabb", the output should be
    solution(inputString) = true.

We can rearrange "aabb" to make "abba", which is a palindrome."""


def palindrome_rearranging(inputString):
    one_single = False
    counter = Counter(inputString)
    for k, v in counter.items():
        if v % 2 != 0:
            if one_single:
                return False
            else:
                one_single = True

    return True


# this code is not correct, b/c the single appearance letters will be repeatly handled
# def palindrome_rearranging(inputString):
# single_count = 0
# for c in inputString:
#     if inputString.count(c)%2!=0:
#         single_count += 1

#     if single_count > 1:
#         return False

# return True


"""Call two arms equally strong if the heaviest weights they each are able 
to lift are equal.

Call two people equally strong if their strongest arms are equally strong 
(the strongest arm can be both the right and the left), and so are their 
weakest arms.

Given your and your friend's arms' lifting capabilities find out if you 
two are equally strong.

Example

For yourLeft = 10, yourRight = 15, friendsLeft = 15, and friendsRight = 10, 
the output should be
    solution(yourLeft, yourRight, friendsLeft, friendsRight) = true;

For yourLeft = 15, yourRight = 10, friendsLeft = 15, and friendsRight = 
10, the output should be
    solution(yourLeft, yourRight, friendsLeft, friendsRight) = true;

For yourLeft = 15, yourRight = 10, friendsLeft = 15, and friendsRight = 9, 
the output should be
    solution(yourLeft, yourRight, friendsLeft, friendsRight) = false."""


def are_equally_strong_1(yourLeft, yourRight, friendsLeft, friendsRight):
    your_strong = max(yourLeft, yourRight)
    your_weak = min(yourLeft, yourRight)
    friend_strong = max(friendsLeft, friendsRight)
    friend_weak = min(friendsLeft, friendsRight)

    return your_strong == friend_strong and your_weak == friend_weak


def are_equally_strong_2(yourLeft, yourRight, friendsLeft, friendsRight):
    return max(yourLeft, yourRight) == max(friendsLeft, friendsRight) and min(
        yourLeft, yourRight
    ) == min(friendsLeft, friendsRight)


"""Given an array of integers, find the maximal absolute difference 
between any two of its adjacent elements.

Example

For inputArray = [2, 4, 1, 0], the output should be
solution(inputArray) = 3."""


def array_max_adj_dif(inputArray):
    dif = 0
    for i in range(len(inputArray) - 1):
        dif = max(dif, abs(inputArray[i] - inputArray[i + 1]))

    return dif


def array_max_adj_dif_2(lst):
    lst1 = lst[1:]
    lst2 = lst[:-1]

    return max([abs(a - b) for a, b in zip(lst1, lst2)])


"""An IP address is a numerical label assigned to each device (e.g., 
computer, printer) participating in a computer network that uses the 
Internet Protocol for communication. There are two versions of the 
Internet protocol, and thus two versions of addresses. One of them is 
the IPv4 address.

Given a string, find out if it satisfies the IPv4 address naming rules.

Example

For inputString = "172.16.254.1", the output should be
    solution(inputString) = true;

For inputString = "172.316.254.1", the output should be
    solution(inputString) = false.

316 is not in range [0, 255].

For inputString = ".254.255.0", the output should be
    solution(inputString) = false.

There is no first number."""


def is_IPv4(inputString):
    lst = inputString.split(".")
    if len(lst) != 4:
        return False

    for n in lst:
        if (
            not n.isdecimal()
            or int(n) not in range(0, 256)
            or len(n) > 1
            and n[:1] == "0"
        ):
            return False

    return True


def is_IPv4(inputString):
    return (
        re.match(
            r"^((2[0-4][0-9]|25[0-5]|1[0-9][0-9]|[1-9][0-9]|[0-9])[.]){3}(2[0-4][0-9]|25[0-5]|1[0-9][0-9]|[1-9][0-9]|[0-9])$",
            inputString,
        )
        != None
    )


"""You are given an array of integers representing coordinates of obstacles 
situated on a straight line.

Assume that you are jumping from the point with coordinate 0 to the right. 
You are allowed only to make jumps of the same length represented by some 
integer.

Find the minimal length of the jump enough to avoid all the obstacles.

Example

For inputArray = [5, 3, 6, 7, 9], the output should be
    solution(inputArray) = 4.

Check out the image below for better understanding:"""


def avoid_obstacles_online(inputArray):
    for i in range(2, 40):
        non_divisible = True
        for j in range(0, len(inputArray)):
            if int(inputArray[j]) % i == 0:
                non_divisible = False
                break

        if non_divisible:
            return i


def avoid_obstacles(inputArray):
    # if not inputArray:
    #     return 2

    for i in range(
        2, max(inputArray) + 2
    ):  # +2 or more to ensure covering the last one
        # divisible means if you use that i (steps) to jump, you will hit one of nbrs in array
        # maybe one jump or a few jumps, you will hit that number, so the i no good
        divisible = False

        for j in range(0, len(inputArray)):
            if inputArray[j] % i == 0:
                divisible = True
                break

        if not divisible:
            return i

    return len(inputArray) + 1


def avoid_obstacles_2(inputArray):
    if not inputArray:
        return 2

    def is_not_divisible(x):
        for n in inputArray:
            print("n in inputArray: ", n)
            if n % x == 0:
                return False

        return True

    return [i for i in range(2, max(inputArray) + 2) if is_not_divisible(i)][0]


def avoid_obstacles_gpt(ary):
    def is_not_divisible(x):
        for n in ary:
            if n % x == 0:
                return False
        return True

    max_val = max(ary)
    for jump_length in range(2, max_val + 2):
        print(f"Testing jump length: {jump_length}")
        if is_not_divisible(jump_length):
            print(f"Found valid jump length: {jump_length}")
            return jump_length

    return -1


# Hypothesis test with timeout decorator
@given(st.lists(st.integers(min_value=1, max_value=10), min_size=1, max_size=10))
def test_js_avoid_obstacles(nums: list[int]):
    assert avoid_obstacles(nums) == avoid_obstacles_2(nums)


"""Last night you partied a little too hard. Now there's a black and white 
photo of you that's about to go viral! You can't let this ruin your reputation, 
so you want to apply the box blur algorithm to the photo to hide its content.

The pixels in the input image are represented as integers. The algorithm 
distorts the input image in the following way: Every pixel x in the output 
image has a value equal to the average value of the pixel values from the 
3 × 3 square that has its center at x, including x itself. All the pixels on 
the border of x are then removed.

Return the blurred image as an integer, with the fractions rounded down.

Example

For

image = [[1, 1, 1], 
         [1, 7, 1], 
         [1, 1, 1]]
the output should be 
    solution(image) = [[1]].

To get the value of the middle pixel in the input 3 × 3 square: (1 + 1 + 1 + 1 
+ 7 + 1 + 1 + 1 + 1) = 15 / 9 = 1.66666 = 1. The border pixels are cropped 
from the final result.

For

image = [[7, 4, 0, 1], 
         [5, 6, 2, 2], 
         [6, 10, 7, 8], 
         [1, 4, 2, 0]]
the output should be
    solution(image) = [[5, 4], 
                       [4, 4]]

There are four 3 × 3 squares in the input image, so there should be four 
integers in the blurred output. To get the first value: (7 + 4 + 0 + 5 + 6 + 
2 + 6 + 10 + 7) = 47 / 9 = 5.2222 = 5. The other three integers are 
obtained the same way, then the surrounding integers are cropped from the 
final result."""


# 23 under island of knowledge
def box_blur_1(image):
    rows = len(image)
    cols = len(image[0])

    res = [[0 for i in range(cols - 2)] for j in range(rows - 2)]

    for i in range(rows - 2):
        for j in range(cols - 2):
            sm = 0
            for k in range(3):
                for m in range(3):
                    sm += image[i + k][j + m]

            res[i][j] = sm // 9

    return res


import numpy as np


def box_blur_2(image):
    image = np.array(image)

    rows = len(image)
    cols = len(image[0])

    res = [[0 for _ in range(cols - 2)] for _ in range(rows - 2)]

    for i in range(rows - 2):
        for j in range(cols - 2):
            sub = image[i : i + 3, j : j + 3]

            res[i][j] = np.sum(sub) // 9

    return res


"""In the popular Minesweeper game you have a board with some mines and 
those cells that don't contain a mine have a number in it that indicates 
the total number of mines in the neighboring cells. Starting off with some 
arrangement of mines we want to create a Minesweeper game setup.

Example

For
matrix = [[true, false, false],
          [false, true, false],
          [false, false, false]]

the output should be
    solution(matrix) = [[1, 2, 1],
                        [2, 1, 1],
                        [1, 1, 1]]"""


def mine_sweeper(matrix):
    rows = len(matrix)
    cols = len(matrix[0])

    # new_matrix = matrix.copy()     #don't use matrix.copy(), the copy will affect original
    new_matrix = [[0 for j in range(cols)] for i in range(rows)]

    for i in range(rows):
        for j in range(cols):
            sm = 0
            if i == 0 and j == 0:
                sm += 1 if matrix[i][j + 1] else 0
                sm += 1 if matrix[i + 1][j] else 0
                sm += 1 if matrix[i + 1][j + 1] else 0
            elif i == 0 and j == cols - 1:
                print(i, j)
                print(matrix[i][j - 1])
                print(sm)
                if matrix[i][j - 1]:
                    sm += 1
                print(sm)
                sm += 1 if matrix[i + 1][j - 1] else 0
                print(sm)
                sm += 1 if matrix[i + 1][j] else 0
                print(sm)
            elif i == rows - 1 and j == 0:
                sm += 1 if matrix[i - 1][j] else 0
                sm += 1 if matrix[i][j + 1] else 0
                sm += 1 if matrix[i - 1][j + 1] else 0
            elif i == rows - 1 and j == cols - 1:
                sm += 1 if matrix[i][j - 1] else 0
                sm += 1 if matrix[i - 1][j] else 0
                sm += 1 if matrix[i - 1][j - 1] else 0
            elif i == 0 and j not in (0, cols - 1):  # 4 sides
                sm += 1 if matrix[i][j - 1] else 0
                sm += 1 if matrix[i][j + 1] else 0
                sm += 1 if matrix[i + 1][j - 1] else 0
                sm += 1 if matrix[i + 1][j + 1] else 0
                sm += 1 if matrix[i + 1][j] else 0
            elif j == 0 and i not in (0, rows - 1):  # left
                sm += 1 if matrix[i - 1][j] else 0
                sm += 1 if matrix[i + 1][j] else 0
                sm += 1 if matrix[i - 1][j + 1] else 0
                sm += 1 if matrix[i][j + 1] else 0
                sm += 1 if matrix[i + 1][j + 1] else 0
            elif i not in (0, rows - 1) and j == cols - 1:  # right
                sm += 1 if matrix[i - 1][j - 1] else 0
                sm += 1 if matrix[i - 1][j] else 0
                sm += 1 if matrix[i][j - 1] else 0
                sm += 1 if matrix[i + 1][j - 1] else 0
                sm += 1 if matrix[i + 1][j] else 0
            elif i == rows - 1 and j not in (0, cols - 1):  # bottom
                sm += 1 if matrix[i - 1][j - 1] else 0
                sm += 1 if matrix[i - 1][j] else 0
                sm += 1 if matrix[i - 1][j + 1] else 0
                sm += 1 if matrix[i][j - 1] else 0
                sm += 1 if matrix[i][j + 1] else 0
            else:
                sm += 1 if matrix[i - 1][j - 1] else 0  # middle
                sm += 1 if matrix[i - 1][j] else 0
                sm += 1 if matrix[i - 1][j + 1] else 0
                sm += 1 if matrix[i][j - 1] else 0
                sm += 1 if matrix[i][j + 1] else 0
                sm += 1 if matrix[i + 1][j - 1] else 0
                sm += 1 if matrix[i + 1][j] else 0
                sm += 1 if matrix[i + 1][j + 1] else 0

            new_matrix[i][j] = sm

    return new_matrix


def mine_sweeper_2(matrix):
    rows = len(matrix)
    cols = len(matrix[0])

    new_matrix = [[0 for _ in range(cols)] for _ in range(rows)]

    for i in range(rows):
        for j in range(cols):
            sm = 0
            # upper left
            row = i - 1
            col = j - 1
            if row in range(rows) and col in range(cols):
                sm += 1 if matrix[row][col] else 0  # middle

            # upper above
            row = i - 1
            col = j
            if row in range(rows) and col in range(cols):
                sm += 1 if matrix[row][col] else 0  # middle

            # upper right
            row = i - 1
            col = j + 1
            if row in range(rows) and col in range(cols):
                sm += 1 if matrix[row][col] else 0  # middle

            # left
            row = i
            col = j - 1
            if row in range(rows) and col in range(cols):
                sm += 1 if matrix[row][col] else 0  # middle

            row = i
            col = j + 1
            if row in range(rows) and col in range(cols):
                sm += 1 if matrix[row][col] else 0  # middle

            row = i + 1
            col = j - 1
            if row in range(rows) and col in range(cols):
                sm += 1 if matrix[row][col] else 0  # middle

            row = i + 1
            col = j
            if row in range(rows) and col in range(cols):
                sm += 1 if matrix[row][col] else 0  # middle

            row = i + 1
            col = j + 1
            if row in range(rows) and col in range(cols):
                sm += 1 if matrix[row][col] else 0  # middle

            new_matrix[i][j] = sm

    return new_matrix


"""Given an array of integers, replace all the occurrences of elemToReplace 
with substitutionElem.

Example

For inputArray = [1, 2, 1], elemToReplace = 1, and substitutionElem = 3, 
the output should be
    solution(inputArray, elemToReplace, substitutionElem) = [3, 2, 3]."""


def array_replace(inputArray, elemToReplace, substitutionElem):
    for k, v in enumerate(inputArray):
        if v == elemToReplace:
            inputArray[k] = substitutionElem

    return inputArray


def array_replace_2(inputArray, elemToReplace, substitutionElem):
    return [substitutionElem if n == elemToReplace else n for n in inputArray]


"""Check if all digits of the given integer are even.

Example
For n = 248622, the output should be
    solution(n) = true;
For n = 642386, the output should be
    solution(n) = false.
"""


def even_digits_only(n):
    for d in list(str(n)):
        if int(d) % 2 != 0:
            return False

    return True


def even_digits_only_2(n):
    return not len([0 for d in list(str(n)) if int(d) % 2 != 0])


def even_digits_only_3(n):
    return len([0 for d in list(str(n)) if int(d) % 2 != 0]) == 0


"""Correct variable names consist only of English letters, digits and 
underscores and they can't start with a digit.

Check if the given string is a correct variable name.

Example
For name = "var_1__Int", the output should be
    solution(name) = true;

For name = "qq-q", the output should be
    solution(name) = false;

For name = "2w2", the output should be
    solution(name) = false.
"""


def variable_name(name):
    return not re.search(r"\W", name) and not re.search(r"^\d", name)


"""Given a string, your task is to replace each of its characters by 
the next one in the English alphabet; i.e. replace a with b, replace b 
with c, etc (z would be replaced by a).

Example
For inputString = "crazy", the output should be 
    solution(inputString) = "dsbaz".
"""


def alphabetic_shift(inputString):
    return "".join([chr(ord(c) + 1) if c != "z" else "a" for c in inputString])


"""Given two cells on the standard chess board, determine whether they have 
the same color or not.

Example
For cell1 = "A1" and cell2 = "C3", the output should be
    solution(cell1, cell2) = true

For cell1 = "A1" and cell2 = "H3", the output should be
    solution(cell1, cell2) = false.    
"""


def chessboard_cell_color(cell1, cell2):
    if cell1 == cell2:
        return True

    dct = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "H": 8}
    # or
    # ltrs = ["A", "B", "C", "D", "E", "F", "G", "H"]
    # nbrs = [1,2,3,4,5,6,7,8]
    # dct = {ltr: nbr for ltr, nbr in zip(ltrs, nbrs)}

    indice1 = [dct[list(cell1)[0]], int(list(cell1)[1])]
    indice2 = [dct[list(cell2)[0]], int(list(cell2)[1])]

    # chess board dif color characteristics: if both x, y are even or odd, color is black
    # if x, y one is even, the other is odd, then color is white
    # before it took me a long time to figure out, i just don't know what to think
    return (
        indice1[0] % 2 == indice1[1] % 2
        and indice2[0] % 2 == indice2[1] % 2
        or indice1[0] % 2 != indice1[1] % 2
        and indice2[0] % 2 != indice2[1] % 2
    )


# just a cleanup from the prev one
def cellboard_cell_color_2(cell1, cell2):
    if cell1 == cell2:
        return True

    dct = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "H": 8}

    ps1 = list(cell1)
    ps2 = list(cell2)

    indice1 = [dct[ps1[0]], int(ps1[1])]
    indice2 = [dct[ps2[0]], int(ps2[1])]

    return (
        indice1[0] % 2 == indice1[1] % 2
        and indice2[0] % 2 == indice2[1] % 2
        or indice1[0] % 2 != indice1[1] % 2
        and indice2[0] % 2 != indice2[1] % 2
    )


"""Consider integer numbers from 0 to n - 1 written down along the circle in 
such a way that the distance between any two neighboring numbers is equal 
(note that 0 and n - 1 are neighboring, too).

Given n and firstNumber, find the number which is written in the radially 
opposite position to firstNumber.

Example
For n = 10 and firstNumber = 2, the output should be
    solution(n, firstNumber) = 7.
"""


# wow. solved like it's nothing. just a few mins ago, i had no clue. still remember had a hard time last year doing it in java
def circle_numbers(n, firstNumber):
    half = n // 2
    return firstNumber + half if firstNumber < half else firstNumber - half


"""You have deposited a specific amount of money into your bank account. Each 
year your balance increases at the same growth rate. With the assumption that 
you don't make any additional deposits, find out how long it would take for 
your balance to pass a specific threshold.

Example

For deposit = 100, rate = 20, and threshold = 170, the output should be
    solution(deposit, rate, threshold) = 3.

Each year the amount of money in your account increases by 20%. So throughout 
the years, your balance would be:

year 0: 100;
year 1: 120;
year 2: 144;
year 3: 172.8.
Thus, it will take 3 years for your balance to pass the threshold, so the 
answer is 3.
"""


def deposti_profit(deposit, rate, threshold):
    if deposit >= threshold:
        return 0

    x = deposit
    yr = 0
    while x < threshold:
        x = x * (1 + rate / 100)
        yr += 1

    return yr


"""
Given a sorted array of integers a, your task is to determine which element 
of a is closest to all other values of a. In other words, find the element 
x in a, which minimizes the following sum:

abs(a[0] - x) + abs(a[1] - x) + ... + abs(a[a.length - 1] - x)
(where abs denotes the absolute value)

If there are several possible answers, output the smallest one.

Example

For a = [2, 4, 7], the output should be 
    solution(a) = 4.

for x = 2, the value will be abs(2 - 2) + abs(4 - 2) + abs(7 - 2) = 7.
for x = 4, the value will be abs(2 - 4) + abs(4 - 4) + abs(7 - 4) = 5.
for x = 7, the value will be abs(2 - 7) + abs(4 - 7) + abs(7 - 7) = 8.

The lowest possible value is when x = 4, so the answer is 4.

For a = [2, 3], the output should be 
    solution(a) = 2.

for x = 2, the value will be abs(2 - 2) + abs(3 - 2) = 1.
for x = 3, the value will be abs(2 - 3) + abs(3 - 3) = 1.
Because there is a tie, the smallest x between x = 2 and x = 3 is the answer.
"""


def absolute_values_sum_minimization(a):
    min_dif = 0

    for i in range(len(a)):
        min_dif += abs(a[i] - a[0])

    elem = a[0]
    for i in range(len(a)):
        dif = 0
        for j in range(len(a)):
            dif += abs(a[i] - a[j])

        if dif < min_dif:
            elem = a[i]
            min_dif = dif

    return elem


"""
Given an array of equal-length strings, you'd like to know if it's possible 
to rearrange the order of the elements in such a way that each consecutive 
pair of strings differ by exactly one character. Return true if it's possible, 
and false if not.

Note: You're only rearranging the order of the strings, not the order of the 
letters within the strings!

Example

For inputArray = ["aba", "bbb", "bab"], the output should be
    solution(inputArray) = false.

There are 6 possible arrangements for these strings:

["aba", "bbb", "bab"]
["aba", "bab", "bbb"]
["bbb", "aba", "bab"]
["bbb", "bab", "aba"]
["bab", "bbb", "aba"]
["bab", "aba", "bbb"]

None of these satisfy the condition of consecutive strings differing by 1 
character, so the answer is false.

For inputArray = ["ab", "bb", "aa"], the output should be
    solution(inputArray) = true.

It's possible to arrange these strings in a way that each consecutive pair 
of strings differ by 1 character (eg: "aa", "ab", "bb" or "bb", "ab", "aa"), 
so return true.
"""


def string_rearrangement(inputArray):
    matrix = []
    permute(inputArray, 0, len(inputArray), matrix)

    # for row in matrix:
    #     print(row)

    # print()
    return compare_strs(matrix)


def permute(lst, l, r, matrix):
    if l == r:
        matrix.append(lst[:])
    else:
        for i in range(l, r):
            swap_words(lst, l, i)
            permute(lst, l + 1, r, matrix)
            swap_words(lst, l, i)


def swap_words(lst, l, r):
    if l == r:
        return

    lst[l], lst[r] = lst[r], lst[l]
    return


# this version passes all tests
# abbreviated version passes all tests too
# if issue (more than 2 difs), immediately discard, if no issue found,
# then it's an instant success
# this is better than _2
def compare_strs(matrix):
    for row in matrix:
        found = True
        # difs = []
        for i in range(len(row) - 1):
            dif = [a == b for a, b in zip(list(row[i]), list(row[i + 1]))]
            if dif.count(False) != 1:
                found = False
                break
            # else:
            #     difs.append(dif[:])

        # if found:
        #     for d in difs:
        #         if d.count(False)!=1:
        #             found = False
        #             break

        if found:  # if no issue found, then those 3 strings are good
            return True

    return False


# this version is much simpler. theorectically a little more inefficient
# but this version can't pass extensive testing of long lists on code signal
def compare_strs_2(matrix):
    for row in matrix:
        found = True
        difs = []
        for i in range(len(row) - 1):
            dif = [a == b for a, b in zip(list(row[i]), list(row[i + 1]))]
            difs.append(dif[:])

        for d in difs:
            if d.count(False) != 1:
                found = False
                break

        if found:
            return True

    return False


"""
Given array of integers, remove each kth element from it.

Example

For inputArray = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] and k = 3, the output should be
    solution(inputArray, k) = [1, 2, 4, 5, 7, 8, 10]
"""


def extract_each_kth(inputArray, k):
    new_arr = []
    for idx, v in enumerate(inputArray):
        if (idx + 1) % k == 0:
            continue
        else:
            new_arr.append(inputArray[idx])

    return new_arr


# online
def extract_each_kth_online(inputArray, k):
    # ew: array slicing arr[start:end:step]. arr[::2] slicing every other element
    del inputArray[
        k - 1 :: k
    ]  # understand now. k-1 is start, : end is omitted so to end, k is step
    return inputArray


# after reading about del list elements online
# doesn't work. [k-1:k] will work to replace from k-1 to k, but with ::,
# it won't work
# def extract_each_kth_2_2 (inputArray, k):
#     inputArray[k-1::k] = []     #replacing elements from k-1 to end with k jump
#     return inputArray


# online
def extract_each_kth_3(inputArray, k):
    return list(
        [x for i, x in enumerate(inputArray) if (i + 1) % k != 0]
    )  # ew change from i%k to (i+1)%k


# chatgpt and me work together after many trials
def extract_each_kth_k_plus_1th_2(nums, k):
    if k == 1:
        return []

    return [
        nums[i] for i in range(len(nums)) if i == 0 or (i + 1) % k != 0 and i % k != 0
    ]


# inputArray = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]==[1, 2, 4, 5, 7, 8, 10]
# inputArray = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]==[1, 2, 5, 8]
def extract_each_kth_k_plus_1th(nums, k):
    n2 = []
    i = 0
    while i < len(nums):
        if (i + 1) % k == 0:
            i += 2
            continue
        else:
            n2.append(nums[i])
            i += 1

    return n2


"""
Find the leftmost digit that occurs in a given string.

Example
For inputString = "var_1__Int", the output should be
    solution(inputString) = '1';
For inputString = "q2q-q", the output should be
    solution(inputString) = '2';
For inputString = "0ss", the output should be
    solution(inputString) = '0'.
"""


def find_1st_digit(inputString):
    return inputString[re.search(r"\d", inputString).start()]


"""
Given a string, find the number of different characters in it.

Example
For s = "cabca", the output should be
    solution(s) = 3.

There are 3 different characters a, b and c.
"""


# find dif letters in a string
def different_symbols_naive(s):
    return len(set(s))


"""
Given array of integers, find the maximal possible sum of some of its k 
consecutive elements.

Example
For inputArray = [2, 3, 5, 1, 6] and k = 2, the output should be
    solution(inputArray, k) = 8.

All possible sums of 2 consecutive elements are:
2 + 3 = 5;
3 + 5 = 8;
5 + 1 = 6;
1 + 6 = 7.
Thus, the answer is 8
"""


# not efficient enough to pass last test when submitting
def array_max_consecutive_sum(inputArray, k):
    mx = 0
    for i in range(len(inputArray) - k + 1):
        mx = max(mx, sum(inputArray[i : i + k]))

    return mx


# sliding window method. passed!
def array_max_consecutive_sum_chatgpt(inputArray, k):
    # if k > (ln := len(inputArray)):
    #    return 0

    # wow. this works too. so they don't matter
    if k >= (ln := len(inputArray)):
        return sum(inputArray)

    sub_sum = sum(inputArray[:k])
    max_sum = sub_sum

    for i in range(ln - k):
        sub_sum = sub_sum - inputArray[i] + inputArray[i + k]
        max_sum = max(sub_sum, max_sum)

    return max_sum


"""
Caring for a plant can be hard work, but since you tend to it regularly, 
you have a plant that grows consistently. Each day, its height increases 
by a fixed amount represented by the integer upSpeed. But due to lack of 
sunlight, the plant decreases in height every night, by an amount represented 
by downSpeed.

Since you grew the plant from a seed, it started at height 0 initially. 
Given an integer desiredHeight, your task is to find how many days it'll 
take for the plant to reach this height.

Example

For upSpeed = 100, downSpeed = 10, and desiredHeight = 910, the output 
should be
    solution(upSpeed, downSpeed, desiredHeight) = 10.

#	Day	Night
1	100	90
2	190	180
3	280	270
4	370	360
5	460	450
6	550	540
7	640	630
8	730	720
9	820	810
10	910	900
The plant first reaches a height of 910 on day 10.
"""


def growing_plant(upSpeed, downSpeed, desiredHeight):
    if upSpeed > desiredHeight:
        return 1

    days = 0
    height = 0
    while height < desiredHeight:
        days += 1
        height += upSpeed
        if height >= desiredHeight:
            return days
        height -= downSpeed

    return days - 1


"""
You found two items in a treasure chest! The first item weighs weight1 and 
is worth value1, and the second item weighs weight2 and is worth value2. 
What is the total maximum value of the items you can take with you, assuming 
that your max weight capacity is maxW and you can't come back for the items 
later?

Note that there are only two items and you can't bring more than one item 
of each type, i.e. you can't take two first items or two second items.

Example

For value1 = 10, weight1 = 5, value2 = 6, weight2 = 4, and maxW = 8, the 
output should be
    solution(value1, weight1, value2, weight2, maxW) = 10.

You can only carry the first item.

For value1 = 10, weight1 = 5, value2 = 6, weight2 = 4, and maxW = 9, the output should be
    solution(value1, weight1, value2, weight2, maxW) = 16.

You're strong enough to take both of the items with you.

For value1 = 5, weight1 = 3, value2 = 7, weight2 = 4, and maxW = 6, the output should be
    solution(value1, weight1, value2, weight2, maxW) = 7.

You can't take both items, but you can take any of them.
"""


def knapsack_light(value1, weight1, value2, weight2, maxW):
    return (
        value1 + value2
        if weight1 + weight2 <= maxW
        else (
            0
            if weight1 > maxW and weight2 > maxW
            else (
                value1
                if value1 >= value2 or weight2 > maxW and weight1 <= maxW
                else (
                    value2
                    if value2 > value1 or weight1 > maxW and weight2 <= maxW
                    else 0
                )
            )
        )
    )


"""
Given a string, output its longest prefix which contains only digits.

Example

For inputString = "123aa1", the output should be
    solution(inputString) = "123".
"""


def longest_digits_prefix(inputString):
    return inputString[: re.search(r"^\d*", inputString).end()]


"""
Let's define digit degree of some positive integer as the number of times 
we need to replace this number with the sum of its digits until we get to 
a one digit number.

Given an integer, find its digit degree.

Example

For n = 5, the output should be
    solution(n) = 0;
For n = 100, the output should be
    solution(n) = 1.
1 + 0 + 0 = 1.
For n = 91, the output should be
    solution(n) = 2.
9 + 1 = 10 -> 1 + 0 = 1.
"""


def digit_degree(n):
    if n < 10:
        return 0

    count = 1
    while len(str(n)) > 1:
        digit_sum = sum([int(s) for s in str(n)])
        if digit_sum < 10:
            return count
        else:
            count += 1
            n = digit_sum

    return count


"""
Given the positions of a white bishop and a black pawn on the standard 
chess board, determine whether the bishop can capture the pawn in one move.

The bishop has no restrictions in distance for each move, but is limited 
to diagonal movement. Check out the example below to see how it can move:

Example

For bishop = "a1" and pawn = "c3", the output should be
    solution(bishop, pawn) = true.

For bishop = "h1" and pawn = "h3", the output should be
    solution(bishop, pawn) = false.
"""


def bishop_and_pawn(bishop, pawn):
    ltrs = ["a", "b", "c", "d", "e", "f", "g", "h"]
    nbrs = [1, 2, 3, 4, 5, 6, 7, 8]
    dct = {ltr: nbr for ltr, nbr in zip(ltrs, nbrs)}

    bishop_indices = [dct[bishop[0]], int(bishop[1])]
    pawn_indices = [dct[pawn[0]], int(pawn[1])]

    return abs(bishop_indices[0] - pawn_indices[0]) == abs(
        bishop_indices[1] - pawn_indices[1]
    )


"""
A string is said to be beautiful if each letter in the string appears 
at most as many times as the previous letter in the alphabet within the 
string; ie: b occurs no more times than a; c occurs no more times than 
b; etc. Note that letter a has no previous letter.

Given a string, check whether it is beautiful.

Example

For inputString = "bbbaacdafe", the output should be 
    solution(inputString) = true.

This string contains 3 as, 3 bs, 1 c, 1 d, 1 e, and 1 f (and 0 of every 
other letter), so since there aren't any letters that appear more 
frequently than the previous letter, this string qualifies as beautiful.

For inputString = "aabbb", the output should be 
    solution(inputString) = false.

Since there are more bs than as, this string is not beautiful.

For inputString = "bbc", the output should be 
    solution(inputString) = false.

Although there are more bs than cs, this string is not beautiful because 
there are no as, so therefore there are more bs than as.
"""


def is_beautiful_str(inputString):
    if len(inputString) == 0:
        return True

    counter = Counter(inputString)
    lst = [(k, v) for k, v in counter.items()]
    lst.sort(key=lambda x: x[0])

    if lst[0][0] != "a":
        return False

    for i in range(len(lst) - 1):
        cur_char, cur_count = lst[i]
        next_char, next_count = lst[i + 1]

        if ord(next_char) - ord(cur_char) > 1 or next_count > cur_count:
            return False

    return True


"""
An email address such as "John.Smith@example.com" is made up of a local 
part ("John.Smith"), an "@" symbol, then a domain part ("example.com").

The domain name part of an email address may only consist of letters, 
digits, hyphens and dots. The local part, however, also allows a lot of 
different special characters. Here you can look at several examples of 
correct and incorrect email addresses.

Given a valid email address, find its domain part.

Example

For address = "prettyandsimple@example.com", the output should be
    solution(address) = "example.com";

For address = "fully-qualified-domain@codesignal.com", the output should be
    solution(address) = "codesignal.com".
"""


def find_email_domain(address):
    return address[address.rfind("@") + 1 :]


"""
Given a string, find the shortest possible string which can be achieved 
by adding characters to the end of initial string to make it a palindrome.

Example

For st = "abcdc", the output should be
    solution(st) = "abcdcba".
"""


def build_palindrome(st):
    if st == st[::-1]:
        return st

    for i in range(len(st)):
        suffix = st[0 : i + 1][::-1]  # got a substring, then reverse
        temp = st + suffix
        if temp == temp[::-1]:
            return temp

    return st


"""
Elections are in progress!

Given an array of the numbers of votes given to each of the candidates 
so far, and an integer k equal to the number of voters who haven't cast 
their vote yet, find the number of candidates who still have a chance to 
win the election.

The winner of the election must secure strictly more votes than any other 
candidate. If two or more candidates receive the same (maximum) number 
of votes, assume there is no winner at all.

Example

For votes = [2, 3, 5, 2] and k = 3, the output should be
    solution(votes, k) = 2.

The first candidate got 2 votes. Even if all of the remaining 3 candidates 
vote for him, he will still have only 5 votes, i.e. the same number as 
the third candidate, so there will be no winner. 
The second candidate can win if all the remaining candidates vote for him 
(3 + 3 = 6 > 5).
The third candidate can win even if none of the remaining candidates vote 
for him. For example, if each of the remaining voters cast their votes 
for each of his opponents, he will still be the winner (the votes array 
will thus be [3, 4, 5, 3]).
The last candidate can't win no matter what (for the same reason as the 
first candidate).
Thus, only 2 candidates can win (the second and the third), which is 
the answer.
"""


def elections_winners(votes, k):
    votes.sort(reverse=True)

    count = 0
    mx = votes[0]
    if k == 0:
        if (cnt := votes.count(mx)) == 1:
            return 1  # 1 winner
        elif cnt > 1:
            return 0  # no winner

    for c in votes:
        if c + k > mx:
            count += 1

    return count


"""
A media access control address (MAC address) is a unique identifier 
assigned to network interfaces for communications on the physical 
network segment.

The standard (IEEE 802) format for printing MAC-48 addresses in 
human-friendly form is six groups of two hexadecimal digits (0 to 9 
or A to F), separated by hyphens (e.g. 01-23-45-67-89-AB).

Your task is to check by given string inputString whether it corresponds 
to MAC-48 address or not.

Example

For inputString = "00-1B-63-84-45-E6", the output should be
    solution(inputString) = true;

For inputString = "Z1-1B-63-84-45-E6", the output should be
    solution(inputString) = false;

For inputString = "not a MAC-48 address", the output should be
    solution(inputString) = false.
"""


def is_MAC48_address_1(inputString):
    res = re.search(
        "^[0-9A-F][0-9A-F]-[0-9A-F][0-9A-F]-[0-9A-F][0-9A-F]-[0-9A-F][0-9A-F]-[0-9A-F][0-9A-F]-[0-9A-F][0-9A-F]$",
        inputString,
    )
    if not res:
        return False

    return True


def is_MAC48_address_2(inputString):
    return re.search("^([0-9A-F][0-9A-F]-){5}[0-9A-F][0-9A-F]$", inputString) != None


def is_MAC48_address_3(inputString):
    return re.match("^([0-9A-F][0-9A-F]-){5}[0-9A-F][0-9A-F]$", inputString) != None


"""
Determine if the given character is a digit or not.

Example

For symbol = '0', the output should be
    solution(symbol) = true;

For symbol = '-', the output should be
    solution(symbol) = false.
"""


def is_digit(symbol):
    return symbol.isdigit()


"""
Given a string, return its encoding defined as follows:

First, the string is divided into the least possible number of disjoint 
substrings consisting of identical characters
for example, "aabbbc" is divided into ["aa", "bbb", "c"]
Next, each substring with length greater than one is replaced with a 
concatenation of its length and the repeating character
for example, substring "bbb" is replaced by "3b"
Finally, all the new strings are concatenated together in the same order 
and a new string is returned.
Example

For s = "aabbbc", the output should be
    solution(s) = "2a3bc".
"""


def line_encoding(s):
    ln = len(s)
    temp = ""
    prev = s[0]
    count = 0
    for k, c in enumerate(s):
        if c == prev:
            count += 1
            if k == ln - 1:
                if count == 1:
                    temp += prev
                else:
                    temp += str(count) + prev
        else:
            if count == 1:
                temp += prev
            else:
                temp += str(count) + prev
            count = 1
            prev = c
            if k == ln - 1:
                temp += c

    return temp


def line_encoding_2(s):
    matches = re.finditer(r"(.)\1*", s)
    temp = ""
    for m in matches:
        sub = m.group()
        ln = len(sub)
        if ln > 1:
            sub = str(ln) + sub[0]
        temp += sub

    return temp


def line_encoding_3(s):
    matches = re.finditer(r"(.)\1*", s)
    temp = ""
    for m in matches:
        sub = m.group()
        sub = str(len(sub)) + sub[0]
        temp += sub

    return re.sub(r"^1(?=[a-z])|(?<=[a-z])1(?=[a-z])", "", temp)


def line_encoding_4(s):
    matches = re.finditer(r"(.)\1*", s)
    temp = ""
    for m in matches:
        sub = m.group()
        ln = len(sub)
        # sub = str(ln if ln>1 else '') + sub[0]
        sub = (str(ln) if ln > 1 else "") + sub[0]
        temp += sub

    print(s)
    return temp


def line_encoding_5(s):
    matches = re.finditer(r"(.)\1*", s)
    temp = ""
    for m in matches:
        ln = len(m.group())
        temp += (str(ln) if ln > 1 else "") + m.group()[0]

    print(s)
    return temp


"""
Given a position of a knight on the standard chessboard, find the number 
of different moves the knight can perform.

The knight can move to a square that is two squares horizontally and one 
square vertically, or two squares vertically and one square horizontally 
away from it. The complete move therefore looks like the letter L. Check 
out the image below to see all valid moves for a knight piece that is 
placed on one of the central squares

Example

For cell = "a1", the output should be
    solution(cell) = 2.

For cell = "c2", the output should be
    solution(cell) = 6.
"""


def chess_knight(cell):
    ltrs = ["a", "b", "c", "d", "e", "f", "g", "h"]
    nbrs = [1, 2, 3, 4, 5, 6, 7, 8]

    dct = {l: n for l, n in zip(ltrs, nbrs)}
    indices = [dct[cell[0]], int(cell[1])]

    count = 0
    if indices[0] + 1 <= 8:
        if indices[1] + 2 <= 8:
            count += 1
        if indices[1] - 2 >= 1:
            count += 1

    if indices[0] + 2 <= 8:
        if indices[1] + 1 <= 8:
            count += 1
        if indices[1] - 1 >= 1:
            count += 1

    if indices[0] - 1 >= 1:
        if indices[1] + 2 <= 8:
            count += 1
        if indices[1] - 2 >= 1:
            count += 1

    if indices[0] - 2 >= 1:
        if indices[1] + 1 <= 8:
            count += 1
        if indices[1] - 1 >= 1:
            count += 1

    return count


"""
Given some integer, find the maximal number you can obtain by deleting 
exactly one digit of the given number.

Example

For n = 152, the output should be
    solution(n) = 52;

For n = 1001, the output should be
    solution(n) = 101.
"""


def delete_digit(n):
    mx = 0
    s = str(n)
    for i in range(len(s)):
        mx = max(mx, int(s[0:i] + s[i + 1 :]))

    return mx


"""
Define a word as a sequence of consecutive English letters. Find the 
longest word from the given string.

Example

For text = "Ready, steady, go!", the output should be
    solution(text) = "steady".
"""


def longest_word(text):
    matches = re.findall(r"\b[a-zA-Z]+\b", text)
    matches.sort(key=lambda s: len(s))
    return matches[-1]


"""
Check if the given string is a correct time representation of the 24-hour 
clock.

Example

For time = "13:58", the output should be
    solution(time) = true;

For time = "25:51", the output should be
    solution(time) = false;

For time = "02:76", the output should be
    solution(time) = false.
"""


def valid_time(time):
    m = re.search(r"([2][0-3]|[0-1][0-9]):[0-5][0-9]", time)
    if m:
        return True
    else:
        return False


"""
CodeMaster has just returned from shopping. He scanned the check of the 
items he bought and gave the resulting string to Ratiorg to figure out 
the total number of purchased items. Since Ratiorg is a bot he is 
definitely going to automate it, so he needs a program that sums up 
all the numbers which appear in the given input.

Help Ratiorg by writing a function that returns the sum of numbers that 
appear in the given inputString.

Example

For inputString = "2 apples, 12 oranges", the output should be
    solution(inputString) = 14.
"""


def sumup_numbers(inputString):
    matches = re.findall(r"\d+", inputString)
    return sum([int(m) for m in matches])


"""Given a rectangular matrix containing only digits, calculate the 
number of different 2 × 2 squares in it.

Example

For
matrix = [[1, 2, 1],
          [2, 2, 2],
          [2, 2, 2],
          [1, 2, 3],
          [2, 2, 1]]
the output should be
    solution(matrix) = 6.

Here are all 6 different 2 × 2 squares:

[1 2
 2 2]
[2 1
 2 2]
[2 2
 2 2]
[2 2
 1 2]
[2 2
 2 3]
[2 3
 2 1]
"""

from itertools import chain


def different_squares(matrix):
    rows = len(matrix)
    cols = len(matrix[0])

    total = (rows - 2 + 1) * (cols - 2 + 1)
    new_matrix = [[[0 for _ in range(2)] for _ in range(2)] for _ in range(total)]

    t = -1
    for i in range(rows - 2 + 1):
        for j in range(cols - 2 + 1):
            t += 1
            for k in range(2):
                for m in range(2):
                    new_matrix[t][k][m] = matrix[i + k][j + m]

    m_set = set()
    for i in range(total):
        m = str(list(chain.from_iterable(new_matrix[i])))
        m_set.add(m)

    return len(m_set)


# don't need to flatten the matrices. to python interpreter, [] or [[]] looks very flat
def different_squares_2(matrix):
    rows = len(matrix)
    cols = len(matrix[0])

    total = (rows - 2 + 1) * (cols - 2 + 1)
    new_matrix = [[[0 for _ in range(2)] for _ in range(2)] for _ in range(total)]

    t = -1
    for i in range(rows - 2 + 1):
        for j in range(cols - 2 + 1):
            t += 1
            for k in range(2):
                for m in range(2):
                    new_matrix[t][k][m] = matrix[i + k][j + m]

    m_set = set()
    for i in range(total):
        m = str(new_matrix[i])

        m_set.add(m)

    return len(m_set)

    # dct = {}

    # for i in range(total):
    #     m = str(list(chain.from_iterable(new_matrix[i])))
    #     if m in dct.keys():
    #         dct[m] = dct[m] + 1
    #     else:
    #         dct[m] = 1

    # return len(dct.keys())


def different_squares_3(matrix):
    rows = len(matrix)
    cols = len(matrix[0])
    total = (rows - 2 + 1) * (cols - 2 + 1)
    new_matrix = [[[0 for _ in range(2)] for _ in range(2)] for _ in range(total)]

    t = -1
    for i in range(rows - 2 + 1):
        for j in range(cols - 2 + 1):
            t += 1
            for k in range(2):
                for m in range(2):
                    new_matrix[t][k][m] = matrix[i + k][j + m]

    m_set = set()
    for m in new_matrix:
        s = str(m)
        m_set.add(s)

    return len(m_set)


def different_squares_4(matrix):
    rows = len(matrix)
    cols = len(matrix[0])
    total = (rows - 2 + 1) * (cols - 2 + 1)
    new_matrix = [[[0 for _ in range(2)] for _ in range(2)] for _ in range(total)]

    t = -1
    for i in range(rows - 2 + 1):
        for j in range(cols - 2 + 1):
            t += 1
            for k in range(2):
                for m in range(2):
                    new_matrix[t][k][m] = matrix[i + k][j + m]

    return len(set([str(m) for m in new_matrix]))


def different_squares_5(matrix):
    rows = len(matrix)
    cols = len(matrix[0])
    #    total = (rows-2+1)*(cols-2+1)

    m_set = set()
    for i in range(rows - 2 + 1):
        for j in range(cols - 2 + 1):
            sub = [[0 for _ in range(2)] for _ in range(2)]
            for k in range(2):
                for m in range(2):
                    sub[k][m] = matrix[i + k][j + m]

            m_set.add(str(sub))

    return len(m_set)


def different_squares_6(matrix):
    rows = len(matrix)
    cols = len(matrix[0])
    arr = np.array(matrix)

    m_set = set()
    for i in range(rows - 2 + 1):
        for j in range(cols - 2 + 1):
            sub = arr[
                i : i + 2, j : j + 2
            ]  # using slicing numpy array to extract subarrays

            m_set.add(str(sub))

    return len(m_set)


"""
Given an integer product, find the smallest positive (i.e. greater than 
0) integer the product of whose digits is equal to product. If there 
is no such integer, return -1 instead.

Example

For product = 12, the output should be
    solution(product) = 26;

For product = 19, the output should be
    solution(product) = -1.
"""


def digitsProduct_online(product):
    answerDigits = []
    answer = 0

    if product == 0:
        return 10

    if product == 1:
        return 1

    for divisor in range(9, 1, -1):
        while product % divisor == 0:
            product /= divisor
            answerDigits.append(divisor)

    if product > 1:
        return -1

    for i in range(len(answerDigits) - 1, -1, -1):
        answer = 10 * answer + answerDigits[i]

    return answer


def digits_product(product):
    if product == 0:
        return 10

    if product == 1:
        return 1

    lst = []
    for i in range(9, 1, -1):
        while product % i == 0:
            lst.append(i)
            product /= i

    if product != 1:
        return -1

    temp = 0
    for i in range(len(lst) - 1, -1, -1):
        temp = temp * 10 + lst[i]

    return temp


def digits_product_2(product):
    if product == 0:
        return 10

    if product == 1:
        return 1

    lst = []
    for i in range(9, 1, -1):  # 9 b/c only have to check digits
        while (
            product % i == 0
        ):  # this is important b/c same factor could appears more than one time. so don't use if
            lst.append(i)
            product /= i

    if product != 1:
        return -1

    lst.reverse()
    return int("".join([str(n) for n in lst]))
    # return int(''.join(str(n) for n in (lst[::-1])))


"""You are given an array of strings names representing filenames. 
The array is sorted in order of file creation, such that names[i] 
represents the name of a file created before names[i+1] and after 
names[i-1] (assume 0-based indexing). Because all files must have 
unique names, files created later with the same name as a file created 
earlier should have an additional (k) suffix in their names, where k is 
the smallest positive integer (starting from 1) that does not appear 
in previous file names.

Your task is to iterate through all elements of names (from left to 
right) and update all filenames based on the above. Return an array 
of proper filenames.

Example

For names = ["doc", "doc", "image", "doc(1)", "doc"], the output should be 
    solution(names) = ["doc", "doc(1)", "image", "doc(1)(1)", "doc(2)"].

Since names[0] = "doc" and names[1] = "doc", update names[1] = "doc(1)"
Since names[1] = "doc(1)" and names[3] = "doc(1)", update names[3] = "doc(1)(1)"
Since names[0] = "doc", names[1] = "doc(1)", and names[4] = "doc", 
update names[4] = "doc(2)"
"""


# after looking at online solution. with my improvements
def file_naming(names):
    for i in range(1, len(names)):
        if names[i] in names[:i]:
            n = 1
            while f"{names[i]}({n})" in names[:i]:
                n += 1
            names[i] = f"{names[i]}({n})"

    return names


# walrus operator! nice
def file_naming_2(names):
    for i in range(1, len(names)):
        if (name := names[i]) in names[:i]:
            n = 1
            while (new_name := f"{name}({n})") in names[:i]:
                n += 1
            names[i] = new_name

    return names


"""You are taking part in an Escape Room challenge designed specifically 
for programmers. In your efforts to find a clue, you've found a binary 
code written on the wall behind a vase, and realized that it must be an 
encrypted message. After some thought, your first guess is that each 
consecutive 8 bits of the code stand for the character with the 
corresponding extended ASCII code.

Assuming that your hunch is correct, decode the message.

Example

For code = "010010000110010101101100011011000110111100100001", the output 
should be
    solution(code) = "Hello!".

The first 8 characters of the code are 01001000, which is 72 in the binary 
numeral system. 72 stands for H in the ASCII-table, so the first letter 
is H. Other letters can be obtained in the same manner.
"""


def message_from_binary_code(code):
    ln = len(code) // 8
    temp = ""
    for i in range(ln):
        ch = chr(int(code[i * 8 : i * 8 + 8], 2))
        temp += ch

    return temp


import re


def message_from_binary_code_2(code):
    matches = re.findall(r"\d{8}", code)
    temp = ""
    for m in matches:
        temp += chr(int(m, 2))

    return temp


"""
Construct a square matrix with a size N × N containing integers from 
1 to N * N in a spiral order, starting from top-left and in clockwise 
direction.

Example

For n = 3, the output should be
solution(n) = [[1, 2, 3],
               [8, 9, 4],
               [7, 6, 5]]
"""


def spiral_numbers(n):
    matrix = [[0 for _ in range(n)] for _ in range(n)]

    top = 0
    bottom = len(matrix) - 1
    left = 0
    right = len(matrix[0]) - 1

    m = 1
    while left <= right and top <= bottom:
        for c in range(left, right + 1):
            matrix[top][c] = m
            m += 1

        top += 1
        for r in range(top, bottom + 1):
            matrix[r][right] = m
            m += 1

        right -= 1
        for c in range(right, left - 1, -1):
            matrix[bottom][c] = m
            m += 1

        bottom -= 1
        for r in range(bottom, top - 1, -1):
            matrix[r][left] = m
            m += 1

        left += 1

    return matrix


"""
Sudoku is a number-placement puzzle. The objective is to fill a 9 × 9 
grid with digits so that each column, each row, and each of the nine 
3 × 3 sub-grids that compose the grid contains all of the digits from 
1 to 9.

This algorithm should check if the given grid of numbers represents 
a correct solution to Sudoku.

Example

For
grid = [[1, 3, 2, 5, 4, 6, 9, 8, 7],
        [4, 6, 5, 8, 7, 9, 3, 2, 1],
        [7, 9, 8, 2, 1, 3, 6, 5, 4],
        [9, 2, 1, 4, 3, 5, 8, 7, 6],
        [3, 5, 4, 7, 6, 8, 2, 1, 9],
        [6, 8, 7, 1, 9, 2, 5, 4, 3],
        [5, 7, 6, 9, 8, 1, 4, 3, 2],
        [2, 4, 3, 6, 5, 7, 1, 9, 8],
        [8, 1, 9, 3, 2, 4, 7, 6, 5]]
the output should be
    solution(grid) = true;

For
grid = [[1, 3, 4, 2, 5, 6, 9, 8, 7],
        [4, 6, 8, 5, 7, 9, 3, 2, 1],
        [7, 9, 2, 8, 1, 3, 6, 5, 4],
        [9, 2, 3, 1, 4, 5, 8, 7, 6],
        [3, 5, 7, 4, 6, 8, 2, 1, 9],
        [6, 8, 1, 7, 9, 2, 5, 4, 3],
        [5, 7, 6, 9, 8, 1, 4, 3, 2],
        [2, 4, 5, 6, 3, 7, 1, 9, 8],
        [8, 1, 9, 3, 2, 4, 7, 6, 5]]
the output should be
    solution(grid) = false.

The output should be false: each of the nine 3 × 3 sub-grids should 
contain all of the digits from 1 to 9.
"""


def Sudoku(grid):
    digit_sm = sum([1, 2, 3, 4, 5, 6, 7, 8, 9])
    for i in range(0, 9, 3):
        for j in range(0, 9, 3):
            sm = 0
            lst: list[int] = []
            for k in range(3):
                for m in range(3):
                    cell_val = grid[i + k][j + m]
                    sm += cell_val
                    lst.append(cell_val)

            if sm != digit_sm:
                return False

            if len(list(set(lst))) != len(lst):
                return False

    for row in grid:
        if sum(row) != digit_sm:
            return False

    for j in range(9):
        col_sm = 0
        for i in range(9):
            col_sm += grid[i][j]

        if col_sm != digit_sm:
            return False

    return True


import numpy as np


def Sudokus_2(grid):
    grid = np.array(grid)

    for i in range(0, 9 - 2, 3):
        for j in range(0, 9 - 2, 3):
            sub = grid[i : i + 3, j : j + 3]
            lst = sub.reshape(-1)
            if not verify(lst):
                return False

    for i in range(9):
        row = grid[i, :]
        col = grid[:, i]
        if not verify(row) or not verify(col):
            return False

    return True


def verify(nums: list[int]):
    if min(nums) != 1 or max(nums) != 9 or len(set(nums)) != 9:
        return False

    return True


def main():
    grid = [
        [1, 3, 2, 5, 4, 6, 9, 8, 7],
        [4, 6, 5, 8, 7, 9, 3, 2, 1],
        [7, 9, 8, 2, 1, 3, 6, 5, 4],
        [9, 2, 1, 4, 3, 5, 8, 7, 6],
        [3, 5, 4, 7, 6, 8, 2, 1, 9],
        [6, 8, 7, 1, 9, 2, 5, 4, 3],
        [5, 7, 6, 9, 8, 1, 4, 3, 2],
        [2, 4, 3, 6, 5, 7, 1, 9, 8],
        [8, 1, 9, 3, 2, 4, 7, 6, 5],
    ]

    print(Sudokus_2(grid))

    # names = ["doc", "doc", "image", "doc(1)", "doc"]
    # print(file_naming_2(names))
    # ["doc", "doc(1)", "image", "doc(1)(1)", "doc(2)"

    # print(is_beautiful_str('bbbaacdafe'))
    # print(is_beautiful_str('zaa'))
    # s = "aaabbddac"
    # matches = re.findall(r'(.)\1*',s)
    # for match in matches:
    #     print (match)

    # s = "faaabbddac"
    # print(line_encoding_5(s))

    # matrix = [[1, 2, 1],
    #       [2, 2, 2],
    #       [2, 2, 2],
    #       [1, 2, 3],
    #       [2, 2, 1]]

    # matrix = [[9,9,9,9,9],
    #     [9,9,9,9,9],
    #     [9,9,9,9,9],
    #     [9,9,9,9,9],
    #     [9,9,9,9,9],
    #     [9,9,9,9,9]]

    # matrix = [[9,9,9,9,9],
    #     [9,9,9,9,9],
    #     [9,9,9,9,9],
    #     [9,9,9,9,9],
    #     [9,9,9,9,9],
    #     [9,9,9,9,1]]
    # 2

    # matrix = [
    #     [9, 9, 9, 9, 9],
    #     [9, 9, 9, 9, 9],
    #     [9, 9, 9, 9, 9],
    #     [9, 9, 9, 9, 9],
    #     [9, 9, 9, 9, 9],
    #     [2, 9, 9, 9, 1],
    # ]
    # # 3

    # # print(different_squares_4(matrix))
    # print(different_squares_6(matrix))

    # lst = [m.group()[0] for m in matches]
    # print(lst)

    # obs = [5, 3, 6, 7, 9]
    # obs = [5, 8, 9, 13, 14]
    # obs = [5]
    # obs = [5, 6, 7, 8, 9]
    # print(avoid_obstacles(obs))
    # print(avoid_obstacles_2(obs))

    # testing hypothesis in code module allow you to print info aiding finding out issues
    # test_js_avoid_obstacles()


#    lst = ["aba",  "abb",  "bbb"]
#    string_rearrangement(lst)

# inputArray = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# assert extract_each_kth_2 (inputArray, 3) == [1, 2, 4, 5, 7, 8, 10]
# inputArray = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# assert extract_each_kth_3 (inputArray, 3) == [1, 2, 4, 5, 7, 8, 10]
# inputArray = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# print (extract_each_kth_2(inputArray, 3))
# inputArray = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# print (extract_each_kth_3(inputArray, 3))
# inputArray = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,13, 14]
# print (extract_each_kth_k_plus_1th_2(inputArray, 3))

# inputArray = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
# print (extract_each_kth_k_plus_1th(inputArray, 3))

# inputArray = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]==[1, 2, 5, 8]

if __name__ == "__main__":
    main()
