import pandas as pd
import numpy as np

'''
GRAPE dataset; - Organized by ID, which is Subject#-Eye
               - Each column contains the list of the VF 
                 data per visit
'''
grape_fu = pd.read_csv("../data/GRAPE_FollowUps.csv")
df = grape_fu.drop(columns = ['Interval Years', 'IOP', 'Corresponding CFP'])

df['values'] = df[[str(i) for i in range(60)]].to_numpy().tolist()
df['ID'] = df['Subject Number'].astype(str) + '-'+ df['Laterality'].astype(str)
grouped = df.groupby(['ID', 'Visit Number'], as_index=False)['values'].agg('first')
out = grouped.pivot(index='ID', columns='Visit Number', values='values')
out.to_pickle('GRAPE.pkl')

'''
UW dataset; - Organized by ID, which is Subject#-Eye
            - Each column contains the list of the VF 
              data per visit
            - Rounds any number to nearest int
            - Any non-seen value (0 dB) is now -1
            - Clipped any value greater than 35 to 35
'''

uw_vf = pd.read_csv("../data/UW_VF_Data.csv")

cropped = uw_vf.drop(uw_vf.columns[[1]], axis = 1)
cropped = cropped.drop(cropped.columns[np.r_[3:21]], axis = 1)
df = cropped.drop(cropped.columns[np.r_[57:165]], axis = 1)
df['Eye'] = df['Eye'].replace({'Right': 'OD', 'Left': 'OS'})
df['ID'] =  df['PatID'].astype(str) + '-'+ df['Eye'].astype(str)
cols = [f'Sens_{i}' for i in range(1, 55)]
df[cols] = df[cols].round(0).astype(int)
df[cols] = df[cols].clip(upper=35)
df[cols] = df[cols].replace(0, -1)

df['values'] = df[cols].to_numpy().tolist()
df.rename(columns={'FieldN': 'Visit Number'}, inplace=True)
grouped = df.groupby(['ID', 'Visit Number'], as_index=False)['values'].agg('first')
out = grouped.pivot(index='ID', columns='Visit Number', values='values')
out.to_pickle('UW.pkl')