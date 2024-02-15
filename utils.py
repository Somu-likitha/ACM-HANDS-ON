# defaults imports
import torch

import numpy as np
from sklearn.metrics import accuracy_score

def get_accuracy(
    model,
    data_generator,
    GPU = torch.device("cpu"),
    n = np.inf,
):
    model.eval()
    with torch.no_grad():
        accs = []
        j = 0
        for batch_x, batch_y in data_generator:
            # x_data.reshape(-1, 28, 28).to(GPU), y_data.to(GPU)
            batch_x, batch_y = batch_x.reshape(-1, 28, 28).to(GPU), batch_y.numpy()
            out = model(batch_x).cpu().numpy()
            # print(batch_y.shape, out.shape)
            acc = accuracy_score(batch_y, model(batch_x).argmax(1).cpu().numpy())
            accs.append(acc*100)
            j += 1
            if j == n: break
    model.train()
    return np.array(accs).mean()