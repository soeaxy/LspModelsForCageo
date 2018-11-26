import pandas as pd
from sklearn.utils.class_weight import compute_class_weight

def data_raw(data):
    target = 'value'
    IDCol = 'ID'
    GeoID = data[IDCol]
    print(data[target].value_counts())
    x_columns = [x for x in data.columns if x not in [target,IDCol,'GRID_CODE']]
    X = data[x_columns]
    y = data[target]
    return X, y, GeoID

if __name__ == "__main__":

        data = pd.read_csv('./data/wanzhou_island.csv')
        X, y, _ = data_raw(data)
        # Test
        print(X.shape)

        
        class_weight = 'balanced'
        label = y

        print([label])
        classes = [0, 1]
        weight = compute_class_weight(class_weight, classes, label)

        print(weight)


