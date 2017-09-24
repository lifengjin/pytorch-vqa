import sys
import torch
import matplotlib; matplotlib.use('agg')
import matplotlib.pyplot as plt
import os
import numpy

def main():
    path = sys.argv[1]
    results = []
    files = os.listdir(path)
    files.sort()
    files.remove('singlelayer_joint.pth')
    legends = ['Baseline', 'LSTM+CNN1', 'CNN3', 'LSTM+CNN3', 'CNN']
    for f in files:
        result = torch.load(os.path.join(path, f))

        val_acc = torch.FloatTensor(result['tracker']['val_acc'])
        val_acc = val_acc.mean(dim=1).numpy()
        results.append(val_acc)

    plt.figure()
    for result in results:
        plt.plot(numpy.arange(0, 50),result)
    plt.legend(legends)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.savefig('results.png')


if __name__ == '__main__':
    main()
