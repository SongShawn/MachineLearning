import pandas as pd
import matplotlib

def plot_bar_by_feature(dataset, feature, size_delta=(0,0)):
    features = dataset.columns.tolist()
    if feature not in features:
        return
    data = {feature:[], 'counts':[]}
    for a in dataset[feature].unique():
        data[feature].append(a)
        data['counts'].append(dataset[dataset[feature] == a].shape[0])
    DFData = pd.DataFrame(data)
    DFData = DFData.sort_values(by=['counts'], ascending=False)
    a = DFData.plot.bar(x=feature, y='counts')
    W,H = a.figure.get_size_inches()
    a.figure.set_size_inches(W+size_delta[0],H+size_delta[1])
    a.figure.show()
    # a.figure.savefig('../images/'+feature)