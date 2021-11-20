import json
import numpy  as np
from scipy.stats import wilcoxon

def get_results(filename):
    f = open(filename)
    data = json.load(f)

    ps = []
    ws = []

    for it in data["iterations"]:

  

        matrix = []
        for fold in it["folds"]:
            matrix.append( [  model["metrics"]["test/acc"] for model in fold ] )

        matrix = np.array( matrix )
        print(matrix)

        x = matrix[:,0].tolist()
        y = matrix[:,2].tolist()
        w, p = wilcoxon(x,y,alternative="two-sided")

        ws.append(w)
        ps.append(p)



    f.close()
    return ps, ws 