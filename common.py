import re

def append_dim(df):
    """Append a new column with dimension information parsed from model name"""
    names = df['name']
    dims = []
    for name in names:
        match = re.search(r'-d\d+', name)
        dim_str = match[0]
        dim = int(dim_str[2:]) # Remove first "-d" and "-" from the matched string
        dims.append(dim)

    df['dim'] = dims

    return df

def append_seg_size(df):
    """
    Append a new column with segment size information parsed from model name
    """
    names = df['name']
    seg_sizes = []
    for name in names:
        match = re.search(r'-seg_size\d+', name)
        ss_str = match[0]
        ss = int(ss_str[9:]) # Remove first "-d" and "-" from the matched string
        seg_sizes.append(ss)

    df['seg_size'] = seg_sizes

    return df

def append_class(df):
    """Append a new column with class information parsed from model name"""
    names = df['name']
    vsa_classes = []
    for name in names:
        match = re.search(r'bsc|cgr\d+', name)
        class_str = match[0]
        vsa_classes.append(class_str)

    df['class'] = vsa_classes
    return df

def append_dp(df):
    """Append a new column with dimension information parsed from model name"""
    names = df['name']
    dps = []
    for name in names:
        match = re.search(r'-dp\d+', name)
        dp_str = match[0]
        dp = int(dp_str[3:]) # Remove first "-d" and "-" from the matched string
        dps.append(dp)

    df['dp'] = dps

    return df
