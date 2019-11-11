import argparse
import os.path as op
import csv
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_wine
import re
import pandas as pd

def check_column_names(df, headers=False, column1=False, column2=False):
    # Providing wine.data's headers in alphabetical order   
    if headers:
        headers = np.sort(df.columns.values)
        print("======== Here are wine.data's headers ======== \n" + str('\n'.join(headers)))
        exit()

    # Handling the case where no column names are provided
    if not column1 and not column2:
        print("Please run --help to see the available options for that script.")
        exit()
    
    # Handling the case where only one column name is provided   
    if not column1 or not column2:
        print("Please provide two column names.")
        exit()
    
    # Turning the two provided column names into a list    
    else:
        headers = df.columns.values
        columns = []
        columns.append(column1)
        columns.append(column2)
        wrong_column = []
        for column in columns:
            # Handling the case where one or both provided columns do not exist in the input file
            if column not in headers:
                wrong_column.append(column)  
        if wrong_column:
            for column in wrong_column:
                print(str(column) + " column name does not exist in wine.data file.")   
            print("The script's -H option will allow you to see the input file's column headers.")
            exit()
        else:
            return column1, column2

def plot_data(df, column1=False, column2=False):
    # Getting the unique value of each grape type
    grapetypes = set(df.target)

    # My bulky way of defining the x and y values for each grape type
    xgrape1 = df.loc[df['target'] == list(grapetypes)[0] , column1]
    ygrape1 = df.loc[df['target'] == list(grapetypes)[0] , column2]
    xgrape2 = df.loc[df['target'] == list(grapetypes)[1] , column1]
    ygrape2 = df.loc[df['target'] == list(grapetypes)[1] , column2]
    xgrape3 = df.loc[df['target'] == list(grapetypes)[2] , column1]
    ygrape3 = df.loc[df['target'] == list(grapetypes)[2] , column2]

    # My bulky way of scattering the three sets of data in one frame
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.scatter(xgrape1, ygrape1, c='r', marker='o', label="Grapetype " + str(list(grapetypes)[0]))
    ax1.scatter(xgrape2, ygrape2, c='g', marker='v', label="Grapetype " + str(list(grapetypes)[1]))
    ax1.scatter(xgrape3, ygrape3, c='b', marker=',', label="Grapetype " + str(list(grapetypes)[2]))
    plt.xlabel('x-axis: ' + str(column1), fontsize=9)
    plt.ylabel('y-axis: ' + str(column2), fontsize=9)
    plt.legend(loc='upper right', fontsize=8)
    ax1.set_title('Correlation analysis between ' + str(column1) + ' and ' + str(column2) 
    + '\n attributes from sklearn wine dataset \n', fontsize=9, fontweight='bold')
    plt.savefig("./" + column1 + "_" + column2 + ".png")
    plt.show()

    ## I could not get the for loop working with plotting in a single frame :(
    # for grapetype in grapetypes:
    #     x = df[loc['target']==list(grapetypes)[0], column1]
    #     y = df[loc['target']==list(grapetypes)[1], column2]
    #     fig = plt.figure()
    #     ax1 = fig.add_subplot(111)
    #     ax1.scatter(x, y)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-H', '--headers', action='store_true',
                        help="shows the input file's column headers")
    parser.add_argument('-c1', '--column1', action='store', dest='column1', type=str,
                        help="perform a data sanity check for the provided column")
    parser.add_argument('-c2', '--column2', action='store', dest='column2', type=str,
                        help="perform a data sanity check for the provided column")
    args = parser.parse_args()
    wine_data = load_wine()
    df = pd.DataFrame(wine_data['data'], columns=wine_data['feature_names'])
    df['target'] = wine_data['target']
    check_column_names(df, headers=args.headers, column1=args.column1, column2=args.column2)
    plot_data(df, column1=args.column1, column2=args.column2)

if __name__ == "__main__":
    main()