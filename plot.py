import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

epochs = 100
buf = 0.5
plt.figure(figsize=(15,5))

for i in [1,2,3]:
  trainLosses_filename = "results/vae_{}_trainLosses.txt".format(i)
  validLosses_filename = "results/vae_{}_validLosses.txt".format(i)

  xs = np.arange(1, epochs+1)
  trainLosses = np.loadtxt(trainLosses_filename)[:epochs]
  validLosses = np.loadtxt(validLosses_filename)[:epochs]
  if i!=1:
    trainLosses = np.log(trainLosses) 
    validLosses = np.log(validLosses)
  subplot = 130+i

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

plt.savefig('results/vae_plots_1.png')
# plt.show()
