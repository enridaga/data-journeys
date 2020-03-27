
import numpy as np
import pandas as pd
def map_per_image(label, predictions):
    """Computes the precision score of one image.

    Parameters
    ----------
    label : string
            The true label of the image
    predictions : list
            A list of predicted elements (order does matter, 5 predictions allowed per image)

    Returns
    -------
    score : double
    """    
    try:
        return 1 / (predictions[:5].index(label) + 1)
    except ValueError:
        return 0.0

def map_per_set(labels, predictions):
    """Computes the average over multiple images.

    Parameters
    ----------
    labels : list
             A list of the true labels. (Only one true label per images allowed!)
    predictions : list of list
             A list of predicted elements (order does matter, 5 predictions allowed per image)

    Returns
    -------
    score : double
    """
    return np.mean([map_per_image(l, p) for l,p in zip(labels, predictions)])

#                   (true, [predictions])
assert map_per_image('x', []) == 0.0
assert map_per_image('x', ['y']) == 0.0
assert map_per_image('x', ['x']) == 1.0
assert map_per_image('x', ['x', 'y', 'z']) == 1.0
assert map_per_image('x', ['y', 'x']) == 0.5
assert map_per_image('x', ['y', 'x', 'x']) == 0.5
assert map_per_image('x', ['y', 'z']) == 0.0
assert map_per_image('x', ['y', 'z', 'x']) == 1/3
assert map_per_image('x', ['y', 'z', 'a', 'b', 'c']) == 0.0
assert map_per_image('x', ['x', 'z', 'a', 'b', 'c']) == 1.0
assert map_per_image('x', ['y', 'z', 'a', 'b', 'x']) == 1/5
assert map_per_image('x', ['y', 'z', 'a', 'b', 'c', 'x']) == 0.0

assert map_per_set(['x'], [['x', 'y']]) == 1.0
assert map_per_set(['x', 'z'], [['x', 'y'], ['x', 'y']]) == 1/2
assert map_per_set(['x', 'z'], [['x', 'y'], ['x', 'y', 'z']]) == 2/3
assert map_per_set(['x', 'z', 'k'], [['x', 'y'], ['x', 'y', 'z'], ['a', 'b', 'c', 'd', 'e']]) == 4/9
train_df = pd.read_csv("../input/train.csv")
train_df.head()
labels = train_df['Id'].values
labels
# 5 most common Id
# sample_pred = train_df['Id'].value_counts().nlargest(5).index.tolist()
sample_pred = ['new_whale', 'w_23a388d', 'w_9b5109b', 'w_9c506f6', 'w_0369a5c']
predictions = [sample_pred for i in range(len(labels))]
sample_pred
map_per_set(labels, predictions)