import numpy as np
import pandas as pd


def extra_address_for_block(train_data):
    """将Address列转化为是否含Block的0,1值，添加HasBlock列
    """
    find_block = np.char.find(np.char.lower(np.array(train_data['Address'], dtype=str)), 'block')
    addresses = np.select([find_block<0, find_block>0, find_block==0], [0, 1, 1])
    train_data['HasBlock'] = addresses
    return train_data

def extra_dates(data):
    """Dates列只留下小时作为特征
    """
    dates = data['Dates']
    data['Year'] = dates.dt.year
    data['Month'] = dates.dt.month
    data['Day'] = dates.dt.day
    data['Hour'] = dates.dt.hour
    return data

def extra_address_for_suffix(train_data):
    train_data['PositionType'] = None
    suffixs = ["AL","AV","BL","CR","CT","DR","EX","HWY","HY","LN","PL","PZ","RD","ST","TR","WY","WAY"]
    suffix_names = ["Alley","Avenue","Boulevard","Circle","Court","Drive","Expressway","Highway","Highway",
                    "Lane","Place","Plaza","Road","Street","Terrace","Way","Way"]
    cross_road = "CrossRoad" # 交叉路口，含有/的
    unkown_road = "Unkown"

    # 设置交叉路口
    train_data.loc[train_data['Address'].str.contains('/'), 'PositionType'] = cross_road

    # 查找缩写并设置
    for a in suffixs:
        train_data.loc[(train_data['Address'].str.contains('/') == False) 
                    & (train_data['Address'].str.contains(' '+a+'$')), 'PositionType'] = a
        
    # 查找全程并设置
    for i,d in enumerate(suffix_names):
        train_data.loc[(train_data['PositionType'].isna())
                    & (train_data['Address'].str.contains(d, case=False)), 'PositionType'] = suffixs[i]
        
    # 无法解析的均设置为Unkown
    train_data.loc[(train_data['PositionType'].isna()), 'PositionType'] = unkown_road
    return train_data