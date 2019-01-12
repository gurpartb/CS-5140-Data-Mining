#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 18:04:27 2019
Attempt at solving birthay paradox problem.
for CS-5140 Data Mining

@date: Januart, 10, 2019
@author: gurpartapbhatti
"""

"""
Consider a domain of size n = 5000.

Problem A: Generate random numbers in the domain [n]
until two hae the same value. How many random trials
did this take? We will use k to represent this value.

Approach A:
1. Create a set.
2. Generate a random number 'x' between 1 and 5000.
3. IF set contains x, store number of trials 'k' it
took to generate a duplicate element. ELSE, store x
in the set, GOTO step 2.
"""

import matplotlib.pyplot as plt
import numpy as np
import time


def experiment(n, m):
    """
    int n is the range
    int m is number of experiments
    """
    K = np.zeros(m, dtype = int)
    for i in range(m):
        K[i] = random_num_twice(n)
    return K


def random_num_twice(n):
    """
    This method takes in a positive int n as a parameter,
    and returns number of trials until two randomly
    generated values over domain[n] are the same
    """
    S = {n}
    S.clear()
    s = np.random.randint(n)
    k = 1
    while s not in S:
        S.add(s)
        k += 1
        s = np.random.randint(n)
    return k


def to_cdf(K):
    """
    K is unsorted array of ints
    """
    K_len = len(K)
    K.sort(axis = 0)
    max_k = np.max(K)
    Y = np.zeros(max_k + 1)
    ki_1 = 0
    for i in range(K_len):
        ki = K[i]
        Y[ki_1:ki] = i/K_len
        ki_1 = ki
    # Y[max_k] = K_len/K_len
    Y[max_k] = 1
    return Y


def expected_val(K):
    """
    K is an array of number.
    This method returns the mean of K.
    sum(K)/len(K), or average val of K.
    """
    return np.mean(K)


def single_experiment():
    n = 5000
    m = 300
    s_t = time.time()
    K = experiment(n, m)
    e_t = time.time()
    T[i] = (e_t - s_t) * 1000
    Y = to_cdf(K)
    X = np.linspace(0, np.max(K), np.max(K) + 1)
    plt.plot(X, Y, label = 'normal_CDF')
    plt.legend(bbox_to_anchor=(.35,1))
    plt.xlabel('number of trial')
    plt.ylabel('cumulutive density')
    plt.show()
    E = expected_val(K)
    print('E[k] = ', E)
    print('dt', (e_t - s_t)*1000)


n = 5000
m_1 = 1000
m_n = 10000
dm = 1000
M = np.arange(m_1, m_n + dm, dm)
T = np.zeros(len(M))
for i in range (len(T)):
    s_t = time.time()
    K = experiment(n, M[i])
    e_t = time.time()
    T[i] = (e_t - s_t) * 1000
plt.plot(M, T, 'k o')
plt.xlabel('number of trials \'m\' ')
plt.ylabel('time in milliseconds')
plt.show()

