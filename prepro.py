import pandas as pd
from keras_preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder


# 유사한 학생
def prepro(data):
    data = data.dropna(subset=['ccd','hk', 'grup_cd', 'rk', 's_avg'], axis=0)

    data = data.drop(['sex', 'yy', 'hcd', 'hired_yy', 'co'], axis=1)

    en = LabelEncoder()
    data['grup_cd'] = en.fit_transform(data['grup_cd'])
    data['ccd'] = en.fit_transform(data['ccd'])
    data['s_avg'] = en.fit_transform(data['s_avg'])
    data['hk'] = en.fit_transform(data['hk'])
    data['rk'] = en.fit_transform(data['rk'])
    data['bzc_cd'] = en.fit_transform(data['bzc_cd'])

    data.astype(int)

    # STID와 GRUP_CD를 기준으로 그룹화하고 S_AVG를 기준으로 내림차순 정렬하여 중복 제거하면서 가장 높은 S_AVG를 가진 행 선택
    result = data.sort_values('s_avg', ascending=False).drop_duplicates(['stid', 'grup_cd']).sort_values('stid')

    # STID 열을 기준으로 그룹화하고 CCD와 GROUP_CD 열의 고유한 값을 리스트로 만듦
    grouped = result.groupby('stid').agg({
        'grup_cd': lambda x: x.tolist(),
        'ccd': lambda x: pd.Series.unique(x).tolist(),
        's_avg': lambda x: x.tolist(),
        'hk': lambda x: x.tolist(),
        'rk': lambda x: x.tolist(),
        'bzc_cd': lambda x: pd.Series.unique(x).tolist()
    })

    data = pd.DataFrame(
        {'grup_cd': grouped['grup_cd'], 'ccd': grouped['ccd'], 's_avg': grouped['s_avg'],
         'hk': grouped['hk'],'rk': grouped['rk'], 'bzc_cd': grouped['bzc_cd']})


    length = round(data['grup_cd'].apply(len).max())
    padded_sequences1 = pad_sequences(data['grup_cd'], maxlen=length, padding='post', truncating='post', value=0,
                                      dtype='int')
    padded_sequences2 = pad_sequences(data['ccd'], maxlen=length, padding='post', truncating='post', value=0,
                                      dtype='int')
    padded_sequences3 = pad_sequences(data['s_avg'], maxlen=length, padding='post', truncating='post', value=0,
                                      dtype='int')
    padded_sequences4 = pad_sequences(data['hk'], maxlen=length, padding='post', truncating='post', value=0,
                                      dtype='int')
    padded_sequences5 = pad_sequences(data['rk'], maxlen=length, padding='post', truncating='post', value=0,
                                      dtype='int')
    padded_sequences6 = pad_sequences(data['bzc_cd'], maxlen=length, padding='post', truncating='post', value=0,
                                      dtype='int')

    # 패딩된 시퀀스를 데이터프레임 열로 할당
    data['grup_cd'] = list(padded_sequences1)
    data['ccd'] = list(padded_sequences2)
    data['s_avg'] = list(padded_sequences3)
    data['hk'] = list(padded_sequences4)
    data['rk'] = list(padded_sequences5)
    data['bzc_cd'] = list(padded_sequences6)

    return data, length


def result(data):
    data = data.dropna(subset=['ccd','hk', 'grup_cd', 'rk', 's_avg','bzc_cd'], axis=0)


    # STID와 GRUP_CD를 기준으로 그룹화하고 S_AVG를 기준으로 내림차순 정렬하여 중복 제거하면서 가장 높은 S_AVG를 가진 행 선택
    result = data.sort_values('s_avg', ascending=False).drop_duplicates(['stid', 'grup_cd']).sort_values('stid')

    # STID 열을 기준으로 그룹화하고 CCD와 GROUP_CD 열의 고유한 값을 리스트로 만듦
    grouped = result.groupby('stid').agg({
        'co': lambda x: x.tolist()[0],
        'bzc_cd': lambda x: x.tolist()[0]
    })

    data = pd.DataFrame(
        {'co': grouped['co'],'bzc_cd': grouped['bzc_cd']})

    return data