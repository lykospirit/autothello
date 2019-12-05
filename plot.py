import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

epochs = 100
buf = 0.5
plt.figure(figsize=(15,5))

model = "seq2seq3"
trainLosses_filename = "results/{}_trainLosses.txt".format(model)
validLosses_filename = "results/{}_validLosses.txt".format(model)

xs = np.arange(1, epochs+1)
trainLosses = np.loadtxt(trainLosses_filename)[:epochs]
validLosses = np.loadtxt(validLosses_filename)[:epochs]
plt.subplot(subplot)
plt.xlim((-buf,epochs+buf))
plt.plot(xs, validLosses, 'r-', xs, trainLosses, 'b-')
plt.hlines(min(validLosses), -buf, epochs, 'r', 'dashed')
plt.text(-9, min(validLosses), "{:.2f}".format(min(validLosses)))
plt.hlines(min(trainLosses), -buf, epochs, 'b', 'dashed')
plt.text(-9, min(trainLosses), "{:.2f}".format(min(trainLosses)))
plt.xlabel('Epochs')
if i==1: plt.ylabel('Loss')
else: plt.ylabel('Log Loss')
plt.legend(['Test','Train'])

plt.savefig("results/{}.png".format(model))
# plt.show()
