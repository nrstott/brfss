import os

import pandas as pd
import tensorflow as tf

__data_dir = os.path.join(os.path.dirname(__file__), 'data')
__variables_dir = os.path.join(__data_dir, 'variables')

_train_headers = [
    'index', '_STATE', 'FMONTH', 'IDATE', 'IMONTH', 'IDAY', 'IYEAR',
    'DISPCODE', 'SEQNO', '_PSU', 'CTELENM1', 'PVTRESD1', 'COLGHOUS',
    'STATERES', 'CELLFON4', 'LADULT', 'NUMADULT', 'NUMMEN', 'NUMWOMEN',
    'CTELNUM1', 'CELLFON5', 'CADULT', 'PVTRESD3', 'CCLGHOUS',
    'CSTATE1', 'LANDLINE', 'HHADULT', 'GENHLTH', 'PHYSHLTH',
    'MENTHLTH', 'POORHLTH', 'HLTHPLN1', 'PERSDOC2', 'MEDCOST',
    'CHECKUP1', 'EXERANY2', 'SLEPTIM1', 'CVDINFR4', 'VDCRHD4',
    'CVDSTRK3', 'ASTHMA3', 'ASTHNOW', 'CHCSCNCR', 'CHCOCNCR',
    'CHCCOPD1', 'HAVARTH3', 'ADDEPEV2', 'CHCKIDNY', 'DIABETE3',
    'DIABAGE2', 'LASTDEN3', 'RMVTETH3', 'SEX', 'MARITAL', 'EDUCA',
    'RENTHOM1', 'NUMHHOL2', 'NUMPHON2', 'CPDEMO1', 'VETERAN3',
    'EMPLOY1', 'CHILDREN', 'INCOME2', 'INTERNET', 'WEIGHT2', 'HEIGHT3',
    'PREGNANT', 'DEAF', 'BLIND', 'DECIDE', 'DIFFWALK', 'DIFFDRES',
    'DIFFALON', 'SMOKE100', 'SMOKDAY2', 'STOPSMK2', 'LASTSMK2',
    'USENOW3', 'ECIGARET', 'ECIGNOW', 'ALCDAY5', 'AVEDRNK2',
    'DRNK3GE5', 'MAXDRNKS', 'FLUSHOT6', 'FLSHTMY2', 'PNEUVAC3',
    'TETANUS', 'FALL12MN', 'FALLINJ2', 'SEATBELT', 'DRNKDRI2',
    'HADMAM', 'HOWLONG', 'HADPAP2', 'LASTPAP2', 'HPVTEST', 'HPLSTTST',
    'HADHYST2', 'PCPSAAD2', 'PCPSADI1', 'PCPSARE1', 'PSATEST1',
    'PSATIME', 'PCPSARS1', 'BLDSTOOL', 'LSTBLDS3', 'HADSIGM3',
    'HADSGCO1', 'LASTSIG3', 'HIVTST6', 'HIVTSTD3', 'HIVRISK4',
    'PDIABTST', 'PREDIAB1', 'INSULIN', 'BLDSUGAR', 'FEETCHK2',
    'DOCTDIAB', 'CHKHEMO3', 'FEETCHK', 'EYEEXAM', 'DIABEYE', 'DIABEDU',
    'PAINACT2', 'QLMENTL2', 'QLSTRES2', 'QLHLTH2', 'MEDICARE',
    'HLTHCVR1', 'DELAYMED', 'DLYOTHER', 'NOCOV121', 'LSTCOVRG',
    'DRVISITS', 'MEDSCOST', 'CARERCVD', 'MEDBILL1', 'MEDADVIC',
    'UNDRSTND', 'WRITTEN', 'CAREGIV1', 'CRGVREL1', 'CRGVLNG1',
    'CRGVHRS1', 'CRGVPRB2', 'CRGVPERS', 'CRGVHOUS', 'CRGVMST2',
    'CRGVEXPT', 'CIMEMLOS', 'CDHOUSE', 'CDASSIST', 'CDHELP',
    'CDSOCIAL', 'CDDISCUS', 'SSBSUGR2', 'SSBFRUT2', 'CALRINFO',
    'MARIJANA', 'USEMRJNA', 'ASTHMAGE', 'ASATTACK', 'ASERVIST',
    'ASDRVIST', 'ASRCHKUP', 'ASACTLIM', 'ASYMPTOM', 'ASNOSLEP',
    'ASTHMED3', 'ASINHALR', 'IMFVPLAC', 'HPVADVC2', 'HPVADSHT',
    'SHINGLE2', 'NUMBURN2', 'CNCRDIFF', 'CNCRAGE', 'CNCRTYP1',
    'CSRVTRT1', 'CSRVDOC1', 'CSRVSUM', 'CSRVRTRN', 'CSRVINST',
    'CSRVINSR', 'CSRVDEIN', 'CSRVCLIN', 'CSRVPAIN', 'CSRVCTL1',
    'PROFEXAM', 'LENGEXAM', 'PCPSADE1', 'PCDMDECN', 'SXORIENT',
    'TRNSGNDR', 'RCSGENDR', 'RCSRLTN2', 'CASTHDX2', 'CASTHNO2',
    'EMTSUPRT', 'LSATISFY', 'QLACTLM2', 'USEEQUIP', 'QSTVER',
    'QSTLANG', 'MSCODE', '_STSTR', '_STRWT', '_RAWRAKE', '_WT2RAKE',
    '_CHISPNC', '_CRACE1', '_CPRACE', '_CLLCPWT', '_DUALUSE',
    '_DUALCOR', '_LLCPWT2', '_LLCPWT', '_RFHLTH', '_PHYS14D',
    '_MENT14D', '_HCVU651', '_TOTINDA', '_MICHD', '_LTASTH1',
    '_CASTHM1', '_ASTHMS1', '_DRDXAR1', '_EXTETH2', '_ALTETH2',
    '_DENVST2', '_PRACE1', '_MRACE1', '_HISPANC', '_RACE', '_RACEG21',
    '_RACEGR3', '_RACE_G1', '_AGEG5YR', '_AGE65YR', '_AGE80', '_AGE_G',
    'HTIN4', 'HTM4', 'WTKG3', '_BMI5', '_BMI5CAT', '_RFBMI5',
    '_CHLDCNT', '_EDUCAG', '_INCOMG', '_SMOKER3', '_RFSMOK3',
    '_ECIGSTS', '_CURECIG', 'DRNKANY5', 'DROCDY3_', '_RFBING5',
    '_DRNKWEK', '_RFDRHV5', '_FLSHOT6', '_PNEUMO2', '_RFSEAT2',
    '_RFSEAT3', '_DRNKDRV', '_RFMAM2Y', '_MAM5021', '_RFPAP33',
    '_RFPSA21', '_RFBLDS3', '_COL10YR', '_HFOB3YR', '_FS5YR',
    '_FOBTFS', '_CRCREC', '_AIDTST3'
]


def get_feature_column_key(feature_column):
    if feature_column in bucketized_columns:
        return feature_column.source_column.key

    return feature_column.key


def build_batch(df, batch_size):
    features = df.sample(batch_size)
    label1 = features.pop('USENOW3')
    label2 = features.pop('ECIGNOW')

    features_dict = {col: features[col].tolist() for col in features}

    return features_dict, label1, label2


def load_train_data(filename):
    def ecignow_transform(x):
        if x < 2:
            return 1
        else:
            return 0

    def convert_weight(v: int):
        return float(v) / 1000

    def convert_height(v: int):
        vstr = str(v)
        if len(vstr) == 3:
            ft = float(vstr[0])
            inches = int(vstr[1:3])
        else:
            ft = float(vstr[0:2])
            inches = float(vstr[2:4])
        return ((ft * 12.) + inches) / 95.  # BRFSS 2016 has max height of 7'11 or 95 inches.

    label_columns = ['ECIGNOW', 'USENOW3']

    df = pd.read_csv(filename, names=_train_headers)
    df = df.drop([x for x in _train_headers if x not in [get_feature_column_key(fc) for fc in
                                                         raw_columns] and x not in label_columns],
                 axis=1)

    df = df.loc[
        df.ECIGNOW.notna() & df.ECIGNOW.notnull() & df.EDUCA.notna() & df.EMPLOY1.notna() &
        df.INCOME2.notna() & df.MARITAL.notna() &
        df.SEX.notna() & df._AGEG5YR.notna() &
        df.USENOW3.notna()]

    df = df.loc[df.WEIGHT2.notna() & df.HEIGHT3.notnull()]
    df.WEIGHT2 = df.WEIGHT2.astype(int)
    # 9999 = refused, 7777 = not sure, > 1000 = metric
    df = df.loc[(df.WEIGHT2 != 9999) & (df.WEIGHT2 != 7777) & (df.WEIGHT2 < 1000)]
    df.WEIGHT2 = df.WEIGHT2.apply(convert_weight)

    df = df.loc[df.HEIGHT3.notna() & df.HEIGHT3.notnull()]
    df.HEIGHT3 = df.HEIGHT3.astype(int)
    df = df.loc[(df.HEIGHT3 != 9999) & (df.HEIGHT3 != 7777) & (df.HEIGHT3 < 1000)]
    df.HEIGHT3 = df.HEIGHT3.apply(convert_height)

    df.EDUCA = df.EDUCA.astype(int)
    df.EMPLOY1 = df.EMPLOY1.astype(int)
    #    df.INTERNET = df.INTERNET.astype(int)
    df.INCOME2 = df.INCOME2.astype(int)
    df.MARITAL = df.MARITAL.astype(int)
    df.SEX = df.SEX.astype(int)
    #   df.VETERAN3 = df.VETERAN3.astype(int).apply(lambda x: 1 if x == 1 else 0)
    df._AGEG5YR = df._AGEG5YR.astype(int)
    df.ECIGNOW = df.ECIGNOW.astype(int)
    df.ECIGNOW = df.ECIGNOW.apply(ecignow_transform)
    df.USENOW3 = df.USENOW3.astype(int).apply(lambda x: 1 if x == 1 or x == 2 else 0)

    return df


def _read_csv(key):
    return pd.read_csv(os.path.join(__variables_dir, '%s.csv' % key.lower()))


def _categorical_column(key, dtype=tf.int64):
    df = _read_csv(key)
    if dtype == tf.int64:
        vocabulary_list = df.value.astype('int64')
    else:
        raise Exception('Unsupported dtype for categorical column')

    return tf.feature_column.categorical_column_with_vocabulary_list(key, vocabulary_list=vocabulary_list, dtype=dtype)


numeric_columns = [
    tf.feature_column.numeric_column('WEIGHT2'),
    tf.feature_column.numeric_column('HEIGHT3'),
]

categorical_columns = [
    _categorical_column('SEX'),
    _categorical_column('EMPLOY1'),
    _categorical_column('INCOME2'),
    #    categorical_column('INTERNET'),
    _categorical_column('MARITAL'),
    _categorical_column('EDUCA'),
    #    tf.feature_column.categorical_column_with_vocabulary_list('VETERAN3', vocabulary_list=[0, 1]),
    _categorical_column('_AGEG5YR')
]

children = tf.feature_column.numeric_column('CHILDREN')

bucketized_columns = [
    tf.feature_column.bucketized_column(children, boundaries=[1, 2, 3, 4])
]

raw_columns = numeric_columns + categorical_columns + bucketized_columns

columns = numeric_columns + [tf.feature_column.indicator_column(x) for x in (categorical_columns + bucketized_columns)]
