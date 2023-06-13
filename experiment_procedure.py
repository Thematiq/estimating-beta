import os
import pandas as pd

from tqdm import tqdm
from sklearn.model_selection import TimeSeriesSplit


def load_dataset(index: str, selector: str = 'Otwarcie') -> pd.DataFrame:
    df_to_merge = []
    dir = f'data/{index}/history'
    bar = tqdm(os.listdir(dir))
    for file in bar:
        company = file.split('.')[0]
        bar.set_description(f'Processing {company}')
        df = pd.read_csv(f'{dir}/{file}', parse_dates=['Data'])
        df[f'{company}_value'] = df[selector]
        df[f'{company}_volume'] = df['Wolumen']

        df = df[[f'{company}_value', f'{company}_volume']]
        df_to_merge.append(df)

    res = pd.concat(df_to_merge, axis=1)
    return res.reset_index()


def clean_dataset(df: pd.DataFrame, min_days: int = (365 * 20),
                  drop_to_common_timeseries: bool = True) -> pd.DataFrame:
    df_present_count = (~df.isna()).sum()
    columns_to_remove = df_present_count[df_present_count < min_days].index
    df = df.drop(columns=columns_to_remove)

    if drop_to_common_timeseries:
        df = df.dropna(axis=0, how='any')

    return df


if __name__ == '__main__':
    df = load_dataset('^spx')
    df2 = clean_dataset(df, drop_to_common_timeseries=False)
    print(df2.info())
