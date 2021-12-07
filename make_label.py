from show_img import return_label
import os

def make_labeling(label_original, label):
    final = [0]*len(label_original)

    for i in label.split('_'):
        final[label_original.index(i)] = 1

    return final




if __name__ == '__main__':
    label_original = return_label()
    make_labeling(label_original, 'NG_BK_OK')