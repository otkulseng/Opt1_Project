import numpy as np
import matplotlib.pyplot as plt
from algoritmer import bfgs
from tensegrity import TensegrityStructure
import tests as TEST
import os

try:
    os.mkdir("Bilder")
except FileExistsError:
    print("Directory already created")


RUNALL = True
################################################################################
if RUNALL:
    print("START LOCALMIN")
    ts =  TEST.LOCALMIN

    x0 = np.arange(3 * len(ts.free_weights))
    res, conv = bfgs(x0, ts.func, ts.grad, Niter=1000, convergence_plot=True)
    fig, ax = ts.plot(res)
    plt.savefig("Bilder/localminpos.pdf") # Used in report.

    x0 = -np.arange(3 * len(ts.free_weights))
    res, conv = bfgs(x0, ts.func, ts.grad, Niter=1000, convergence_plot=True)
    fig, ax = ts.plot(res)
    plt.savefig("Bilder/localminneg.pdf") # Used in report.

    print("END LOCALMIN")
################################################################################¨


################################################################################
if RUNALL:
    print("START TEST P25")
    ts =  TEST.P25
    x0 = np.arange(3 * len(ts.free_weights))
    res, conv = bfgs(x0, ts.func, ts.grad, Niter=1000, convergence_plot=True)
    plt.figure()
    plt.plot(np.log10(conv))
    plt.savefig("Bilder/P25conv.pdf") # Used in report.
    fig, ax = ts.plot(res)
    plt.savefig("Bilder/P25.pdf") # Used in report.
    print("END TEST P25")
################################################################################


################################################################################
if RUNALL:
    print("START TEST P69")
    ts =  TEST.P69
    x0 = np.arange(3 * len(ts.free_weights))
    res, conv = bfgs(x0, ts.func, ts.grad, Niter=1000, convergence_plot=True)
    plt.figure()
    plt.plot(np.log10(conv))
    plt.savefig("Bilder/P69conv.pdf") # Used in report.
    fig, ax = ts.plot(res)
    plt.savefig("Bilder/P69.pdf") # Used in report.
    print("END TEST P69")
################################################################################


################################################################################
if RUNALL:
    print("START SANITYCHECK")
    ts = TEST.SANITYCHECK
    x0 = np.arange(3 * len(ts.free_weights))

    mu = 1
    prev = x0
    for i in range(10):
        res, num = bfgs(prev, ts.func(mu), ts.grad(mu), Niter=1000, return_iteration=True)
        print(f' {i}: {num}, {mu} ')

        mu *= 1.5
        if num < 1000:
            mu *= 2
        if num < 500:
            mu *= 2
        if num < 250:
            mu *= 2

        mu = min(mu, 1e10)

        if np.linalg.norm(res - prev) < 1e-12:
            break
        prev = res

    res, conv = bfgs(res, ts.func(mu), ts.grad(mu), Niter=1000, convergence_plot=True)
    plt.figure()
    plt.plot(np.log10(conv))
    plt.savefig("Bilder/sanitycheck.pdf")
    fig, ax = ts.plot(res)
    plt.savefig("Bilder/sanitycheck.pdf")
    print("END SANITYCHECK")
################################################################################

################################################################################
if RUNALL:
    print("START FREESTANDING")
    ts = TEST.FREESTANDING
    x0 = np.arange(3 * len(ts.free_weights))

    mu = 1
    prev = x0
    for i in range(10):
        res, num = bfgs(prev, ts.func(mu), ts.grad(mu), Niter=1000, return_iteration=True)
        print(f' {i}: {num}, {mu} ')

        mu *= 1.5
        if num < 1000:
            mu *= 2
        if num < 500:
            mu *= 2
        if num < 250:
            mu *= 2

        mu = min(mu, 1e10)

        if np.linalg.norm(res - prev) < 1e-12:
            break
        prev = res

    res, conv = bfgs(res, ts.func(mu), ts.grad(mu), Niter=1000, convergence_plot=True)

    plt.figure()
    plt.plot(np.log10(conv))
    plt.savefig("Bilder/freestandingconv.pdf") # Used in report.
    fig, ax = ts.plot(res)
    plt.savefig("Bilder/freestanding.pdf") # Used in report.
    print("END FREESTANDING")
################################################################################

################################################################################
if RUNALL:
    print("START FREESTANDING2")
    n = 2
    ts = TEST.STORIES(n)
    x0 = np.arange(3 * len(ts.free_weights))

    mu = 1
    prev = x0
    for i in range(10):
        res, num = bfgs(prev, ts.func(mu), ts.grad(mu), Niter=1000, return_iteration=True)
        print(f' {i}: {num}, {mu} ')

        mu *= 1.5
        if num < 1000:
            mu *= 2
        if num < 500:
            mu *= 2
        if num < 250:
            mu *= 2

        mu = min(mu, 1e10)

        if np.linalg.norm(res - prev) < 1e-12:
            break

        prev = res

    res, conv = bfgs(res, ts.func(mu), ts.grad(mu), Niter=1000, convergence_plot=True)
    plt.figure()
    plt.plot(np.log10(conv))
    plt.savefig(f'Bilder/{n}freestandingconv.pdf') # Used in report.
    fig, ax = ts.plot(res)
    plt.savefig(f'Bilder/{n}freestanding.pdf') # Used in report.
    print("END FREESTANDING2")
################################################################################


