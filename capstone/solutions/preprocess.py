import numpy as np
import pandas as pd

suffixs = ["AL","AV","BL","CR","CT","DR","EX","HWY","HY","LN","PL","PZ","RD","ST","TR","TER","WY","WAY"]
suffix_names = ["Alley","Avenue","Boulevard","Circle","Court","Drive","Expressway","Highway","Highway",
                    "Lane","Place","Plaza","Road","Street","Terrace","Terrace","Way","Way"]

# 解析过程中无法识别的特征值均设置为该字段
unkown_value = "Unkown"


def extra_address_for_block(train_data):
    """将Address列转化为是否含Block的0,1值，添加HasBlock列
    """
    find_block = np.char.find(np.char.lower(np.array(train_data['Address'], dtype=str)), 'block')
    addresses = np.select([find_block<0, find_block>0, find_block==0], [0, 1, 1])
    train_data['HasBlock'] = addresses
    return train_data

def extra_dates(train_data):
    """Dates列只留下小时作为特征
    """
    dates = train_data['Dates']
    train_data['Year'] = dates.dt.year
    train_data['Month'] = dates.dt.month
    train_data['Day'] = dates.dt.day
    train_data['Hour'] = dates.dt.hour
    return train_data

def extra_address_for_suffix(train_data):
    train_data['RoadType'] = None
    cross_road = "CrossRoad" # 交叉路口，含有/的

    # 设置交叉路口
    train_data.loc[train_data['Address'].str.contains('/'), 'RoadType'] = cross_road

    # 查找缩写并设置
    for a in suffixs:
        train_data.loc[(train_data['Address'].str.contains('/') == False) 
                    & (train_data['Address'].str.contains(' '+a+'$')), 'RoadType'] = a
        
    # 查找全称并设置
    for i,d in enumerate(suffix_names):
        train_data.loc[(train_data['RoadType'].isna())
                    & (train_data['Address'].str.contains(d, case=False)), 'RoadType'] = suffixs[i]
        
    # 无法解析的均设置为Unkown
    train_data.loc[(train_data['RoadType'].isna()), 'RoadType'] = unkown_value

    # 合并 HWY HY，合并 WY WAY
    train_data.loc[train_data['RoadType'] == "HWY", "RoadType"] = "HY"
    train_data.loc[train_data['RoadType'] == "WAY", "RoadType"] = "WY"
    train_data.loc[train_data['RoadType'] == "TER", "RoadType"] = "TR"
    return train_data

def extra_address_for_infos(train_data):
    """
    从Address字段解析出道路编号，道路名称。
    解析不出的，赋值为unkown_value
    """
    col_block = "RoadBlock"
    col_name1 = "RoadName1"
    col_name2 = "RoadName2"
    train_data[col_name1] = None
    train_data[col_name2] = None
    train_data[col_block] = None

    # 1200 Block of 3RD ST
    block_pattern = r'(^\d{1,}) Block'

    # 1200 Block of 3RD ST
    name_patterns1 = [r"of (.*?) "+ i + r"$" for i in suffixs]

    # 5TH ST / MARKET ST 前半段
    name_patterns2 = [r"(.*?) "+ i + r" /" for i in suffixs] 

    # 5TH ST / MARKET ST 后半段
    name_patterns3 = [r"/ (.*?) " + i + r"$" for i in suffixs]

    global recode_row_no
    recode_row_no = 0
    blocks = train_data["Address"].str.findall(block_pattern)
    train_data[col_block] = blocks.map(_block_map)

    recode_row_no=0
    address1 = train_data['Address'].str.findall(name_patterns1[0])
    for i in name_patterns1[1:]:
        address1 += train_data['Address'].str.findall(i)
    address1 = address1.map(_address_map)

    recode_row_no=0
    address2 = train_data["Address"].str.findall(name_patterns2[0])
    for i in name_patterns2[1:]:
        address2 += train_data['Address'].str.findall(i)
    address2 = address2.map(_address_map)

    recode_row_no=0
    address3 = train_data["Address"].str.findall(name_patterns3[0])
    for i in name_patterns3[1:]:
        address3 += train_data['Address'].str.findall(i)
    address3 = address3.map(_address_map)

    # address1有地址的行，address2应该没有地址
    tmp = address2[address1.isna() == False].isna()
    if not tmp.all():
        print("Both address1 and address2 contain road name. index: ", tmp[tmp==False].index.tolist())
        
    # address1有地址的行，address3应该没有地址
    tmp = address3[address1.isna() == False].isna()
    if not tmp.all():
        print("Both address1 and address3 contain road name. index: ", tmp[tmp==False].index.tolist())

    # merge address1 and address2
    tmp = address1.isna()
    address1[tmp] = address2[tmp]

    train_data[col_name1] = address1
    train_data[col_name2] = address3

    # 交叉路口形式只解析出一个道路名称的行索引
    tmp = train_data.loc[(train_data[col_block].isna()) & (train_data[col_name1].isna() | \
            train_data[col_name2].isna()), ["Address", col_block, col_name1, col_name2]].index
    if len(tmp) != 0:
        print("There is only one road name in CrossRoad. index: ", tmp.tolist())

    train_data.loc[train_data[col_block].isna(), col_block] = unkown_value
    train_data.loc[train_data[col_name1].isna(), col_name1] = unkown_value
    train_data.loc[train_data[col_name2].isna(), col_name2] = unkown_value
    return train_data

# 用于打印错误信息
record_row_no = 0

def _block_map(a):
    global record_row_no
    record_row_no += 1
    if not isinstance(a, list):
        raise TypeError("_, row: {}".format(record_row_no-1))
        
    if len(a) != 0:
        if len(a) > 1:
            print("INFO: _block_map row: {} cnt: {} data: {}.".format(record_row_no-1, len(a), a))
        return a[0]
    else:
        return None


def _address_map(a):
    global record_row_no
    record_row_no += 1
    if not isinstance(a, list):
        raise TypeError("_address_map, row: {}".format(record_row_no-1))

    if len(a) != 0:
        if len(a) > 1:
            print("INFO: _address_map row: {} cnt: {} data: {}".format(record_row_no-1, len(a), a))
        return a[0]
    else:
        return None


def dataset_sample(train_data, frac=0.1, feature_name="Category"):
    """
    从数据集中按指定类别采样，如果某类别样本数*frac小于等于10，则进行全采样。
    """
    new_data = pd.DataFrame(columns=train_data.columns)
    feature_values_cnt = train_data[feature_name].value_counts()
    for name, cnt in feature_values_cnt.items():
        if cnt*frac > 10:
            new_data = new_data.append(train_data[train_data[feature_name]==name].sample(frac=frac), ignore_index=True)
        else :
            new_data = new_data.append(train_data[train_data[feature_name]==name], ignore_index=True)

    return new_data.astype(train_data.dtypes.to_dict())
