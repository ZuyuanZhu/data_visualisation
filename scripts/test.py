#!/usr/bin/env python

i = 1
j = 1
for idx in range(10):

    print('i=%d, j=%d' % (i, j))
    if j == i:
        i = i + 1
        j = 1
    elif j < i:
        j = j + 1
