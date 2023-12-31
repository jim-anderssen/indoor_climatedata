#!/Users/jimanderssen/anaconda3/bin/python
#Miran 1925 20.11.2023
import sys
import os
#print(sys.executable)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import statsmodels
import statsmodels.api as sm
import statsmodels.formula.api as smf
# plots a line given an intercept and a slope
from statsmodels.graphics.regressionplots import abline_plot

def parse_from_file_into_csv():
    # Make a new file with only specific columns
    # Data has already been cleaned a little
    # This case: P4_L1, P4_L5
    #These columns P4_L1_Vai.C&% Temp [C];P4_L1_Vai.C&% Temp [C];P4_L1_Vai.C&% Hum [%RH];P4_L1_Vai.C&% Dew [C]
    
    df = pd.read_csv('one year test.csv', sep=';', encoding='latin-1', on_bad_lines='skip')
    #print(df.columns.tolist())
    df = df[['Tid','P4_L5_C&% Temp [C]','P4_L5_C&% Hum [%RH]','P4_L1_Vai.C&% Temp1 [C]','P4_L1_Vai.C&% Temp2 [C]','P4_L1_Vai.C&% Hum [%RH]','P4_L1_Vai.C&% Dew [C]']]
    df.to_csv('L1L5.csv', sep=';', encoding='latin-1')
    print('successful')
    #return df

def parse_weather_into_csv():
    df = pd.read_csv('weather.csv', sep=';', encoding='latin-1', on_bad_lines='skip')
    #print(df.columns.tolist())
    # P4_U.LnsiC& Temp [C];P4_U.LnsiC& Hum [%RH];P4_U.LnsiC& Pres [mBar]
    #O3_U.It_C&% Temp [C];O3_U.It_C&% Hum [%RH];O3_U.It_C&% Pres [mBar]
    df = df[['Tid','P4_U.LnsiC& Temp [C]','O3_U.It_C&% Temp [C]','P4_U.LnsiC& Hum [%RH]','O3_U.It_C&% Hum [%RH]','O3_U.It_C&% Pres [mBar]']]
    df.to_csv('weather.csv', sep=';', encoding='latin-1')
    print('successful')
    #return df

def make_new_weather_file(dataframe):
    print('Making a new weather file')
    #df = pd.read_csv('weather.csv', sep=';', encoding='latin-1', on_bad_lines='skip')
    filename = input('Type in name of file (0 to exit): ')
    if filename == '0':
        return "Returned without making file"

    numeric = dataframe.iloc[:, 2:].select_dtypes(include='number').columns
    dataframe[numeric] = dataframe[numeric].round(2)
    dataframe.to_csv(filename, sep=';', encoding='latin-1', index=False)
    print('successful')

def load_file():
    df = pd.read_csv('one year test.csv', sep=';', encoding='latin-1', on_bad_lines='skip')

    ## Choosing columns from whole df
    data = "'Tid;P4_L10_C&% Temp [C];P4_L10_C&% Hum [%RH];P4_L10_C&% Pres [mBar];P4_L9_C&% Temp [C];P4_L9_C&% Hum [%RH];P4_L9_C&% Pres [mBar];P4_L6_C&% Temp [C];P4_L6_C&% Hum [%RH];P4_L6_C&% Pres [mBar];P4_L4_C&% Temp [C];P4_L4_C&% Hum [%RH];P4_L4_C&% Pres [mBar];P3_L13_C&% Temp [C];P3_L13_C&% Hum [%RH];P3_L13_C&% Pres [mBar];P3_L14_C&% Temp [C];P3_L14_C&% Hum [%RH];P3_L14_C&% Pres [mBar];P4_L8_C&% Temp [C];P4_L8_C&% Hum [%RH];P4_L8_C&% Pres [mBar];P4_L5_C&% Temp [C];P4_L5_C&% Hum [%RH];P4_L5_C&% Pres [mBar];P4_L12_Vai.C& Temp1 [C];P4_L12_Vai.C& Temp2 [C];P4_L12_Vai.C& Hum [%RH];P4_L12_Vai.C& Dew [C];P4_L12_Vai.C& Abs Hum [g/m3];P4_L12_Vai.C& Vapor P [hPa];P4_L1_Vai.C&% Temp1 [C];P4_L1_Vai.C&% Temp2 [C];P4_L1_Vai.C&% Hum [%RH];P4_L1_Vai.C&% Dew [C];P4_L1_Vai.C&% Abs Hum [g/m3];P4_L1_Vai.C&% Vapor P [hPa];P3_L13_Vai.C& Temp1 [C];P3_L13_Vai.C& Temp2 [C];P3_L13_Vai.C& Hum [%RH];P3_L13_Vai.C& Dew [C];P3_L13_Vai.C& Abs Hum [g/m3];P3_L13_Vai.C& Vapor P [hPa];P4_U.LnsiC& Temp [C];P4_U.LnsiC& Hum [%RH];P4_U.LnsiC& Pres [mBar];O3_U.It_C&% Temp [C];O3_U.It_C&% Hum [%RH];O3_U.It_C&% Pres [mBar];P4_L10_Vai.C& Temp1 [C];P4_L10_Vai.C& Temp2 [C];P4_L10_Vai.C& Hum [%RH];P4_L10_Vai.C& Dew [C];P4_L10_Vai.C& Abs Hum [g/m3];P4_L10_Vai.C& Vapor P [hPa];P3_L13_S/YP.Pa Pres [Pa];P4_L7_S/YP.Pa Pres [Pa];P4_Sis/Ulk.Pa Pres [Pa];P4_L2_S/YP.Pa Pres [Pa];P4_S.Ln_C&% Temp [C];P4_S.Ln_C&% Hum [%RH];P4_S.Ln_C&% Pres [mBar];P4_S.It_C&% Temp [C];P4_S.It_C&% Hum [%RH];P4_S.It_C&% Pres [mBar]"
    
    data = data.replace(';', "','" )
    data += "'"
    data_list = [col.strip("'") for col in data.split(',')]
   # print(len(data_list))
    
    df = df[data_list]

    # P3_L13_T;P3_L13_H;P3_L13_P == > S3_L13_T;S3_L13_H;S3_L13_P
    # Making shorter names
    short_col = "'Tid;P4_L10_T;P4_L10_H;P4_L10_P;P4_L9_T;P4_L9_H;P4_L9_P;P4_L6_T;P4_L6_H;P4_L6_P;P4_L4_T;P4_L4_H;P4_L4_P;P3_S13_T;P3_S13_H;P3_S13_P;P3_L14_T;P3_L14_H;P3_L14_P;P4_L8_T;P4_L8_H;P4_L8_P;P4_L5_T;P4_L5_H;P4_L5_P;P4_L12_T1;P4_L12_T2;P4_L12_H;P4_L12_D;P4_L12_AH;P4_L12_VP;P4_L1_T1;P4_L1_T2;P4_L1_H;P4_L1_D;P4_L1_AH;P4_L1_VP;P3_L13_T1;P3_L13_T2;P3_L13_H;P3_L13_D;P3_L13_AH;P3_L13_VP;P4_UL_T;P4_UL_H;P4_UL_P;O3_UI_T;O3_UI_H;O3_UI_P;P4_M10_T1;P4_M10_T2;P4_M10_H;P4_M10_D;P4_M10_AH;P4_M10_VP;P3_L13_S_L;P4_L7_S_L;P4_S_U;P4_L2_S_L;P4_SL_T;P4_SL_H;P4_SL_P;P4_SI_T;P4_SI_H;P4_SI_P"
    short_col = short_col.replace(';', "','" )
    short_col += "'"
    short_list = [col.strip("'") for col in short_col.split(',')]
  #  print(len(short_list))
#    for i in range(len(data_list)):
 #       print(data_list[i])
 #       print(short_list[i])
    # Changing names from long to short in dataFrame
    col_dict = {data_list[i]: short_list[i] for i in range(len(data_list))}
    df.rename(columns=col_dict, inplace=True)
    
    ###

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_colwidth', 20)
    #pd.set_option('display.max_columns', None)
    
    ###

    # Changing to numeric => float
    df = df.apply(lambda col: col.str.replace(',', '.', regex=False) if col.dtype == 'object' else col)
    first_column = df.iloc[:, 0]
    numeric_cols = df.iloc[:,1:].apply(pd.to_numeric, errors='coerce')
    df = pd.concat([first_column, numeric_cols], axis=1)
    
    numeric_cols = df.select_dtypes(include=['float64']).columns
    df[numeric_cols] = df[numeric_cols].astype('float32')
    return df

def load_mean():
    filename = 'mean_data_all.csv'
    print(filename)
    df = pd.read_csv(filename, sep=';', encoding='latin-1', on_bad_lines='skip')
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_colwidth', 20)
    return df
    pass



def shorten_file(df):
    # If you want to shorten the dataset
    df = df[::14]
    return df
    
def calculate_dew(df, sensor):
    #Calculate dew point for climate 
    # T(dt) = (243.04*(ln(RH/100))+(17.625*Ta)/(243.04+Ta))/(17.625-ln(RH/100))
    RH = sensor
    
    Ta = f"{sensor[:-1]+'T'}"    
    ##
    N = (np.log(df[RH]/100)+((17.27*df[Ta])/(237.3+df[Ta])))/17.27
    Dt = (237.3*N)/(1-N)
    ##
    df.insert(df.columns.get_loc(f"{RH}"), f"{sensor[:-1] + 'D'}", Dt)
    return df

def daily_mean(df):
    ##  Make daily mean version of file
    #   Convert tid column to datetime format
    df['Tid'] = pd.to_datetime(df['Tid'], format='%d.%m.%Y %H:%M:%S')

    ##  Extract only the date part from the tid column
    df['Date'] = df['Tid'].dt.date
    daily_mean = df.groupby('Date').mean()
    daily_mean.to_csv('mean_data_all.csv', sep=';', encoding='latin-1')
    #df['Tid'] = df['Tid'].dt.date
    #daily_mean = df.groupby('Tid').mean()
    
    #daily_mean = df.groupby('Date')
    df = daily_mean
    #df.set_index('Tid', inplace=True)

    return df
    

def group_data(df):
    df['Tid'] = pd.to_datetime(df['Tid'], format='%d.%m.%Y %H:%M:%S')
    #df.set_index('Tid', inplace=True)
    return df
    pass

def choose_condition(df):
    # Making different masks
    while True:
        print('Are you using mean or full data?')
        print('P4: L1 (Vai), L4, L5, L6, L8, L9, L10, L12 (Vai) \n'
              'P3: L13 (Vai), L14')    
        sensor = input('Type which sensor (Type 0 to quit): ')
        count_times = 0
        if sensor == '0':
            break
        selection = input('Choose condition (rh, multi = m, blank (=winter)): ')
        if selection == 'rh':
            RH(df, sensor)
        if selection == 'condense':
            df = group_data(df)
            condense(df, sensor)
        if selection == 'm':
            multi_regression(df, sensor)
        #if selection == 'winter':
        #    df_slice = df.loc['2022-10-1':'2023-03-30']
        #    mean_conditions(df_slice, sensor, count_times)
        else:
            mean_conditions(df, sensor, count_times)
            #continue
            """
            if count_times == 0:
                df = daily_mean(df)
            RH_mask = abs(100 - df[sensor] <= 5.0)
            col_to_mask = df.filter(like=f"{sensor[0:6]}").columns
            col_to_mask = list(col_to_mask)
            working_df = df[col_to_mask + ['P4_UL_T','P4_UL_H','P4_UL_P']]
          
            if working_df.shape[1] == 6:
                working_df = calculate_dew(working_df, sensor)
                # Dew mask trial
                dew_mask = (abs(working_df[f"{sensor[:-1]+'T'}"] - working_df[f"{sensor[:-1]+'D'}"]) <= 1.0)
                working_df = working_df[dew_mask]
            else:
                #working_df = df.loc[RH_mask, col_to_mask]
                dew_mask = (abs(working_df[f"{sensor[:-1]+'T1'}"] - working_df[f"{sensor[:-1]+'D'}"]) <= 1.0)
                #working_df = working_df[RH_mask | dew_mask]
                working_df = working_df[dew_mask]
            print(apply_RH_mask(working_df))
            #return df.loc[RH_mask, col_to_mask]
            """

def mean_conditions(df, sensor, count_times):
        #print(len(df))
        #print(df.shape)
        #print(df)

        print('Are you using mean or full data?')
        if len(df) > 365:
            if count_times == 0:
                df = daily_mean(df)
                count_times += 1
        
        #RH_mask = abs(100 - df[sensor] <= 5.0)
        col_to_mask = df.filter(like=f"{sensor[0:6]}").columns
        col_to_mask = list(col_to_mask)
        working_df = df[col_to_mask + ['P4_UL_T','P4_UL_H','P4_UL_P','O3_UI_T', 'O3_UI_H', 'O3_UI_P','P3_L13_S_L','P4_L7_S_L', 'P4_S_U', 'P4_L2_S_L']]
        #working_df = df[col_to_mask + ['P4_UL_T','P4_UL_H','P4_UL_P','O3_UI_T', 'O3_UI_H', 'O3_UI_P']]
        choice = input('Select choice (mask or not)')
        if choice == 'mask':
            if len(sensor) == 7 and sensor[4] != '1':
            #if working_df.shape[1] == 9:
                if count_times == 0:
                    working_df = calculate_dew(working_df, sensor)
                # Dew mask trial
                dew_mask = (abs(working_df[f"{sensor[:-1]+'T'}"] - working_df[f"{sensor[:-1]+'D'}"]) <= 1.5)
                working_df = working_df[dew_mask]

                working_df = working_df.loc['2022-11-01':'2023-03-01']

                masked_df = index_for_delta_weather(working_df, df, sensor)
                print()
                print('Values for masked df:')
                print(apply_RH_mask(masked_df))
                return masked_df

            else:
                dew_mask = (abs(working_df[f"{sensor[:-1]+'T1'}"] - working_df[f"{sensor[:-1]+'D'}"]) <= 1.5)
                #working_df = working_df[RH_mask | dew_mask]
                working_df = working_df[dew_mask]
                
                working_df = working_df.loc['2022-11-01':'2023-03-01']
                
                masked_df = index_for_delta_weather(working_df, df, sensor)
                print()
                print('Values for masked df:')
                print(apply_RH_mask(masked_df))
                return working_df
        else:
            if len(sensor) == 7 and sensor[4] != '1':
                df = calculate_dew(df, sensor)
            df.index = pd.to_datetime(df.index)
            df_slice = df.loc['2022-11-01':'2023-03-01']   
            #print(len(df_slice))
            #print(df_slice.shape)
            #print(df_slice)
            df_slice = mean_delta_weather(df_slice, sensor)
            #print(apply_RH_mask(df_slice))
            include = input('Include specific days (y/n): ')
            if include == 'y':
                masked_df = mean_conditions(df, sensor, count_times)
                print()
                #print('Values for df slice:')
                #print(df_slice.loc[:,'P4_L4_D'])
                #print(apply_RH_mask(df_slice))
                plot_multiple(df_slice, masked_df, sensor)
            elif include == 'n':
                plot_data(df_slice)
            else:
                return  
        return count_times

def multi_regression(df, sensor):
    if len(sensor) == 7 and sensor[4] != '1':
        df = calculate_dew(df, sensor)
    col_to_mask = df.filter(like=f"{sensor[0:6]}").columns
    col_to_mask = list(col_to_mask)
    #P4_L7_S/L;P4_S/U;P4_L2_S/L
    working_df = df[col_to_mask + ['P4_UL_T','P4_UL_H','P4_UL_P','O3_UI_T', 'O3_UI_H', 'O3_UI_P','P3_L13_S_L','P4_L7_S_L', 'P4_S_U', 'P4_L2_S_L','P4_SL_T','P4_SL_H','P4_SL_P','P4_SI_T','P4_SI_H','P4_SI_P']]
    working_df.index = pd.to_datetime(df['Date'])
    #working_df = df[col_to_mask + ['P4_UL_T','P4_UL_H','P4_UL_P','O3_UI_T', 'O3_UI_H', 'O3_UI_P']]
    
    choice = input('Select choice (multi or not)')
    if choice == 'm':
        working_df = mean_delta_weather(working_df, sensor)
        working_df = working_df.loc['2022-10-1':'2023-03-30']   
        working_df.dropna(inplace=True)
        print(len(working_df))
        print(working_df.shape)
        print(working_df.head())
        #fit = smf.ols(f'{sensor} ~ {sensor[:-1]+"T1"} + P4_UL_T + P4_UL_H + P4_UL_P + P4_L7_S_L + P4_L2_S_L + P4_S_U + P4_SL_T + P4_SL_H + P4_SL_P + P4_SI_T + P4_SI_H + P4_SI_P', data = working_df).fit()
        #fit = smf.ols(f'{sensor} ~ {sensor[:-1]+"T"} + P4_UL_T + P4_UL_H + P4_UL_P + P4_L7_S_L + P4_L2_S_L + P4_S_U + P4_SL_T + P4_SL_H + P4_SL_P + P4_SI_T + P4_SI_H + P4_SI_P', data = working_df).fit()
        fit = smf.ols(f'{sensor} ~ P4_L2_S_L', data = working_df).fit()

        #fit = smf.ols(f'{sensor} ~ {sensor[:-1]+"T"}', data = working_df).fit()
        #fit = smf.ols(f'TDew ~ {sensor[:-1]+"T1"}', data = working_df).fit()
        #fit = smf.ols(f'TDew ~ {sensor}', data = working_df).fit()
        #fit = smf.ols(f'{sensor[:-1]+"T"} ~ P4_UL_T + P4_UL_H + P4_UL_P + P4_L7_S_L + P4_L2_S_L + P4_S_U + P4_SL_H + P4_SL_P + P4_SI_T + P4_SI_H + P4_SI_P', data = working_df).fit()
        #fit = smf.ols(f'{sensor[:-1]+"T1"} ~ P4_UL_T + P4_UL_H + P4_UL_P + P4_L7_S_L + P4_L2_S_L + P4_S_U + P4_SL_H + P4_SL_P + P4_SI_T + P4_SI_H + P4_SI_P', data = working_df).fit()
        print(fit.summary())
    pass

def index_for_delta_weather(working_df, original_df, sensor):
    working_df.index = pd.to_datetime(working_df.index)
    
    original_df.index = pd.to_datetime(original_df.index)
    orig_index = original_df.index
    for i, ind in enumerate(working_df.index):
        #print(f"Index {i}: Working_df index: {ind}, Present in Original_df index: {ind in orig_index}")
        #print(f"Working_df index type: {type(ind)}, Original_df index type: {type(orig_index[0])}")
        
        if ind in orig_index:
            if i >= 0:
                prev_ind = ind - pd.Timedelta(days=1)
                if prev_ind in orig_index:
                    #if original_df.loc[ind, 'O3_UI_T'] <= original_df.loc[prev_ind, 'O3_UI_T']:
                    delta_u_temp = original_df.loc[ind, 'O3_UI_T'] - original_df.loc[prev_ind, 'O3_UI_T']
                    working_df.loc[ind, 'DUT'] = delta_u_temp
                    #print(working_df.shape)
                    #if working_df.shape[1] == 11: 
                    if len(sensor) == 7 and sensor[4] != '1':
                        delta_l_temp = original_df.loc[ind, f"{sensor[:-1]+'T'}"] - original_df.loc[prev_ind, f"{sensor[:-1]+'T'}"]
                        working_df.loc[ind, 'DLT'] = delta_l_temp
                        delta_t_dew = working_df.loc[ind, f"{sensor[:-1]+'T'}"] - working_df.loc[ind, f"{sensor[:-1]+'D'}"]
                        working_df.loc[ind, 'TDew'] = delta_t_dew
                    else:
                        delta_l_temp = original_df.loc[ind, f"{sensor[:-1]+'T1'}"] - original_df.loc[prev_ind, f"{sensor[:-1]+'T1'}"]
                        working_df.loc[ind, 'DLT'] = delta_l_temp
                        delta_t_dew = original_df.loc[ind, f"{sensor[:-1]+'T1'}"] - original_df.loc[ind, f"{sensor[:-1]+'D'}"]
                        working_df.loc[ind, 'TDew'] = delta_t_dew
                    #elif original_df.loc[ind, 'O3_UI_T'] > original_df.loc[prev_ind, 'O3_UI_T']:
                     #   delta_temp = original_df.loc[ind, 'O3_UI_T'] - original_df.loc[prev_ind, 'O3_UI_T']
                    #delta_temp = original_df.loc[ind, 'O3_UI_T'].diff()
                else:
                    working_df.loc[ind, 'Delta T'] = pd.NA
            else:
                working_df.loc[ind, 'Delta T'] = pd.NA
        else:
            working_df.loc[ind, 'Delta T'] = pd.NA

    return working_df
#working_df.insert(working_df.columns.get_loc(f"{RH}"), "Delta T", )

def mean_delta_weather(df, sensor):
    df.index = pd.to_datetime(df.index)
    pd.set_option('mode.chained_assignment', None)
    df_copy = df.copy()
    for i, ind in enumerate(df.index):
        #print(f"Index {i}: Working_df index: {ind}, Present in Original_df index: {ind in orig_index}")
        #print(f"Working_df index type: {type(ind)}, Original_df index type: {type(orig_index[0])}")
            if i > 0:
                prev_ind = ind - pd.Timedelta(days=1)
                #if prev_ind in orig_index:
                    #if original_df.loc[ind, 'O3_UI_T'] <= original_df.loc[prev_ind, 'O3_UI_T']:
                delta_u_temp = df.loc[ind, 'O3_UI_T'] - df.loc[prev_ind, 'O3_UI_T']
                df_copy.loc[ind, 'DUT'] = delta_u_temp
                #print(working_df.shape)
                #if working_df.shape[1] == 11: 
                if len(sensor) == 7 and sensor[4] != '1':
                    delta_l_temp = df.loc[ind, f"{sensor[:-1]+'T'}"] - df.loc[prev_ind, f"{sensor[:-1]+'T'}"]
                    df_copy.loc[ind, 'DLT'] = delta_l_temp
                    delta_t_dew = df.loc[ind, f"{sensor[:-1]+'T'}"] - df.loc[ind, f"{sensor[:-1]+'D'}"]
                    df_copy.loc[ind, 'TDew'] = delta_t_dew
                else:
                    delta_l_temp = df.loc[ind, f"{sensor[:-1]+'T1'}"] - df.loc[prev_ind, f"{sensor[:-1]+'T1'}"]
                    df_copy.loc[ind, 'DLT'] = delta_l_temp
                    delta_t_dew = df.loc[ind, f"{sensor[:-1]+'T1'}"] - df.loc[ind, f"{sensor[:-1]+'D'}"]
                    df_copy.loc[ind, 'TDew'] = delta_t_dew
                    #elif original_df.loc[ind, 'O3_UI_T'] > original_df.loc[prev_ind, 'O3_UI_T']:
                     #   delta_temp = original_df.loc[ind, 'O3_UI_T'] - original_df.loc[prev_ind, 'O3_UI_T']
                    #delta_temp = original_df.loc[ind, 'O3_UI_T'].diff()
            
            else:
                df.loc[ind, 'DUT'] = pd.NA

    return df_copy


def RH(df, sensor):
    RH_mask = abs(100 - df[sensor] <= 0.50)
    col_to_mask = df.filter(like=f"{sensor[0:6]}").columns
    col_to_mask = list(col_to_mask)
    working_df = df[['Tid'] + col_to_mask + ['P4_UL_T','P4_UL_H','P4_UL_P']]
    #working_df = df.loc[RH_mask, col_to_mask]
    
    
    if working_df.shape[1] == 7:
        working_df = calculate_dew(working_df, sensor)
        # Dew mask trial
        dew_mask = (abs(working_df[f"{sensor[:-1]+'T'}"] - working_df[f"{sensor[:-1]+'D'}"]) <= 0.5)
        working_df = working_df[RH_mask | dew_mask]
    else:
        #working_df = df.loc[RH_mask, col_to_mask]
        dew_mask = (abs(working_df[f"{sensor[:-1]+'T1'}"] - working_df[f"{sensor[:-1]+'D'}"]) <= 0.5)
        working_df = working_df[RH_mask | dew_mask]
    print(apply_RH_mask(working_df))
    #return df.loc[RH_mask, col_to_mask]

def condense(df, sensor):
    
    col_to_mask = df.filter(like=f"{sensor[:-1]}").columns
    col_to_mask = list(col_to_mask)
    working_df = df[col_to_mask + ['P4_UL_T','P4_UL_H','P4_UL_P']]
    if working_df.shape[1] == 7:
        """
        pass
        condition = (
            (abs(working_df[f"{sensor[:-1]+'T'}"] - working_df[f"{sensor[:-1]+'D'}"]) <= 1.0) &
            (abs(100 - df[sensor] <= 20)) &
            (working_df['P4_UL_T'].diff() < 0) &
            (working_df[f"{sensor[:-1]+'T'}"].diff() > 0) &
            (working_df[f"{sensor[:-1]+'AH'}"].diff() < 0)
        )
        """
    else:

        condition = (
            (abs(working_df[f"{sensor[:-1]+'T1'}"] - working_df[f"{sensor[:-1]+'D'}"]) <= 1.0) &
            (abs(100 - df[sensor] <= 20)) &
            (working_df['P4_UL_T'].diff() < 0) &
            (working_df[f"{sensor[:-1]+'T1'}"].diff() > 0) &
            (working_df[f"{sensor[:-1]+'AH'}"].diff() < 0)
        )
        #'P4_L1_AH'
    #'P4_L1_T1'
        working_df = working_df.sort_values(by='Tid') 
        
        # Filter data based on conditions
        phenomenon = working_df[condition]
        #print(phenomenon)
        indices = phenomenon.index
        print('indices', indices)
        rows_to_cap = []
        window_size = 10
        """
        """
        for index in indices:
            start = max(0, index-5)
            end = min(len(working_df), index + 6)
            rows_to_cap.extend(range(start, end))
        
        rows_to_cap = list(set(rows_to_cap))
        selected_rows = working_df.iloc[rows_to_cap]
        selected_rows = (selected_rows.sort_values(by='Tid'))
        print(selected_rows)
        #print(f"Total number of days: {num_days}")
        counter = 0
        for index in indices:

            start = max(0, index - window_size +1)
            end = min(len(working_df), index +1) 

            window_data = working_df.iloc[start:end]

            if (
                (working_df.loc[start, f"{sensor[:-1]+'AH'}"]) - (working_df.loc[end-1, f"{sensor[:-1]+'AH'}"]) >= 3.0 
                #(working_df.loc[start, f"{sensor}"]) - (working_df.loc[end-1, f"{sensor}"]) <= 0.3 and
                #(working_df.loc[start, f"{'P4_UL_T'}"]) - (working_df.loc[end-1, f"{'P4_UL_T'}"]) <= 0.5
            ):
                

                condition_count = sum(condition.iloc[start:end])
                if condition_count >= 3:
                    rows_to_cap.extend(range(start, end))
                    print(f"Date: Start: {working_df.loc[start, 'Tid']}, End: {working_df.loc[end, 'Tid']}")
                    print(f"{working_df.loc[start, sensor[:-1]+'AH']:.3f}")
                    print(f"{working_df.loc[end-1, sensor[:-1]+'AH']:.3f}")
                    print(f"{working_df.loc[start, sensor[:-1]+'AH'] - working_df.loc[end-1, sensor[:-1]+'AH']:.3f}")
                    counter += 1
                    print(counter)

        rows_to_cap = list(set(rows_to_cap))
        #print(rows_to_cap)
        print('length', len(rows_to_cap))
        
        final_selected_rows = working_df.iloc[rows_to_cap].sort_values(by='Tid')
        unique_dates = final_selected_rows['Tid'].dt.date.unique()
        final_selected_rows.set_index('Tid', inplace=True)
        num_days = len(unique_dates)

        print(final_selected_rows)
        print(f"Total number of days: {num_days}")


def apply_RH_mask(df):
    #print(len(df))
    print('Mean: ')
    print(df.mean())
    print('Std_dev: ')
    print(df.std())
    print(f"Number of days: {len(df)}")
    #return 
    return df

def plot_data(df):
    df.dropna(inplace=True)
    fig, axes = plt.subplots(nrows=1,ncols=2)
    #axes[0].scatter(df['Delta UT'], df['Delta T-Dew'])
    axes[0].plot(df['Delta T-Dew'])
    #axes[0].set_xlabel('D UT')
    axes[0].set_ylabel('D T-Dew')
    axes[0].grid(True)
    

    axes[1].scatter(df['Delta UT'], df['Delta LT'])
    axes[1].set_xlabel('D UT')
    axes[1].set_ylabel('D LT')
    slope, intercept = np.polyfit(df['Delta UT'], df['Delta LT'], 1)
    
    axes[1].plot(df['Delta UT'], slope * df['Delta UT'] + intercept, color='red', label = f"y = {slope:.2f}x + {intercept:.2f}")
    axes[1].legend()
    axes[1].grid(True)
    plt.show(block=False)
    return df

def plot_multiple(df_slice, masked_df, sensor):
    df_slice.dropna(inplace=True)
    #print(masked_df)
    #masked_df.dropna(inplace=True)

    #fig, axes = plt.subplots(nrows=1,ncols=1)
    #axes[0].scatter(df['Delta UT'], df['Delta T-Dew'])
    plt.figure()
    plt.plot(df_slice['TDew'])
    x_min, x_max = plt.gca().get_xlim()
    #print('xmin, xmax', x_min, x_max)
    filtered_masked = masked_df[(masked_df.index >= df_slice.index[0]) & (masked_df.index <= df_slice.index[-1])]
    print(filtered_masked.shape)
    print(filtered_masked)
    plt.scatter(filtered_masked.index, filtered_masked['TDew'], color='red', s=15)
    #plt.scatter(masked_df.index, masked_df['TDew'], color='red', s=15)
    #axes[0].set_xlabel('D UT')
    plt.ylabel('D T-Dew')
    plt.grid(True)    
    plt.text(0.95,0.98, f"Total days: {masked_df['TDew'].count()}", ha='right', va='top', transform=plt.gca().transAxes)
    #Calculate variance in x-axis (days)
    diff_masked = pd.Series(masked_df.index).diff().dt.days.dropna()
    mean_distance = diff_masked.mean()
    #var_masked = diff_masked.var()
    #std_masked = diff_masked.std()
    #pe_std = std_masked/np.sqrt(len(masked_df))
    plt.text(0.95,0.93, f"Mean distance (days): {mean_distance:.2f}", ha='right', va='top', transform=plt.gca().transAxes)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    if len(sensor) == 7:
        plt.title(f"{sensor[0:5]}")
    elif len(sensor) > 7:
        plt.title(f"{sensor[0:6]}")
    #axes[1].scatter(df['DUT'], df['DLT'])
    #axes[1].set_xlabel('D UT')
    #axes[1].set_ylabel('D LT')
    #slope, intercept = np.polyfit(df['DUT'], df['DLT'], 1)
    
    #axes[1].plot(df['DUT'], slope * df['DUT'] + intercept, color='red', label = f"y = {slope:.2f}x")
    #axes[1].legend()
    #axes[1].grid(True)
    """
    if len(sensor) == 7 and sensor[4] != '1':
        axes[2].plot(df[{sensor[:-1]+'T'}])
        axes[2].set_ylabel('T')
        axes[2].grid(True)
    else:
        axes[2].plot(df[{sensor[:-1]+'T1'}])
        axes[2].set_ylabel('T')
        axes[2].grid(True)
    """
    plt.show(block=False)
    return df_slice


def weather():

    df = pd.read_csv('weather.csv', sep=';', encoding='latin-1', on_bad_lines='skip')
    #pd.set_option('display.max_rows', None)
    pd.set_option('display.max_colwidth', 20)
    #df = load_df.copy()

    # Creating shorter variable names: 'P4_U.LnsiC& Temp [C]','O3_U.It_C&% Temp [C]','P4_U.LnsiC& Hum [%RH]','O3_U.It_C&% Hum [%RH]','O3_U.It_C&% Pres [mBar]'
    
    L_T= 'P4_U.LnsiC& Temp [C]'
    I_T = 'O3_U.It_C&% Temp [C]'
    I_H = 'O3_U.It_C&% Hum [%RH]'
    L_H = 'P4_U.LnsiC& Hum [%RH]'
    I_P = 'O3_U.It_C&% Pres [mBar]'

    # Changing to numeric => float
    df[L_T] = pd.to_numeric(df[L_T].str.replace(',', '.', regex=True), errors='coerce')
    df[I_T] = pd.to_numeric(df[I_T].str.replace(',', '.', regex=True), errors='coerce')
    df[I_H] = pd.to_numeric(df[I_H].str.replace(',', '.', regex=True), errors='coerce')
    df[L_H] = pd.to_numeric(df[L_H].str.replace(',', '.', regex=True), errors='coerce')
    df[I_P] = pd.to_numeric(df[I_P].str.replace(',', '.', regex=True), errors='coerce')
    

    
    df['Tid'] = pd.to_datetime(df['Tid'], format='%d.%m.%Y %H:%M:%S')

    ##  Extract only the date part from the tid column
    df['Hour'] = df['Tid'].dt.hour
    df['Date'] = df['Tid'].dt.date
    #weather_columns = df.iloc[:,2:]
    weather_columns = ['P4_U.LnsiC& Temp [C]','O3_U.It_C&% Temp [C]','P4_U.LnsiC& Hum [%RH]','O3_U.It_C&% Hum [%RH]','O3_U.It_C&% Pres [mBar]']
    selected = ['Date','Hour'] + weather_columns
    #return weather_columns

    hourly_mean = df.groupby(['Date', 'Hour']).agg({col: 'mean' for col in weather_columns}).reset_index()
    
    make_new_weather_file(hourly_mean)


    return hourly_mean


    

    pass



def main():

    df = load_file()
    #df = load_mean()
    #df = daily_mean(df)
    current_df = (choose_condition(df))


if __name__ == "__main__":
    main()
    


    ## Every column data = "Tid;P4_S.It_C&% Temp [C];P4_S.It_C&% Hum [%RH];P4_S.It_C&% Pres [mBar];P4_L10_C&% Temp [C];P4_L10_C&% Hum [%RH];P4_L10_C&% Pres [mBar];P4_L9_C&% Temp [C];P4_L9_C&% Hum [%RH];P4_L9_C&% Pres [mBar];P4_L6_C&% Temp [C];P4_L6_C&% Hum [%RH];P4_L6_C&% Pres [mBar];P4_U.LnsiC& Temp [C];P4_U.LnsiC& Hum [%RH];P4_U.LnsiC& Pres [mBar];P4_S.Ln_C&% Temp [C];P4_S.Ln_C&% Hum [%RH];P4_S.Ln_C&% Pres [mBar];P4_L4_C&% Temp [C];P4_L4_C&% Hum [%RH];P4_L4_C&% Pres [mBar];O3_U.It_C&% Temp [C];O3_U.It_C&% Hum [%RH];O3_U.It_C&% Pres [mBar];P3_L13_C&% Temp [C];P3_L13_C&% Hum [%RH];P3_L13_C&% Pres [mBar];P3_L14_C&% Temp [C];P3_L14_C&% Hum [%RH];P3_L14_C&% Pres [mBar];P4_L8_C&% Temp [C];P4_L8_C&% Hum [%RH];P4_L8_C&% Pres [mBar];P4_L5_C&% Temp [C];P4_L5_C&% Hum [%RH];P4_L5_C&% Pres [mBar];P4_L7_S/YP.Pa Pres [Pa];P4_Sis/Ulk.Pa Pres [Pa];P3_L13_S/YP.Pa Pres [Pa];P4_L2_S/YP.Pa Pres [Pa];P4_L12_Vai.C& Temp [C];P4_L12_Vai.C& Temp [C];P4_L12_Vai.C& Hum [%RH];P4_L12_Vai.C& Dew [C];P4_L12_Vai.C& Abs Hum [g/m3];P4_L12_Vai.C& Vapor P [hPa];P4_L1_Vai.C&% Temp1 [C];P4_L1_Vai.C&% Temp2 [C];P4_L1_Vai.C&% Hum [%RH];P4_L1_Vai.C&% Dew [C];P4_L1_Vai.C&% Abs Hum [g/m3];P4_L1_Vai.C&% Vapor P [hPa];P3_L13_Vai.C& Temp [C];P3_L13_Vai.C& Temp [C];P3_L13_Vai.C& Hum [%RH];P3_L13_Vai.C& Dew [C];P3_L13_Vai.C& Abs Hum [g/m3];P3_L13_Vai.C& Vapor P [hPa];P4_L10_Vai.C& Temp1 [C];P4_L10_Vai.C& Temp2 [C];P4_L10_Vai.C& Hum [%RH];P4_L10_Vai.C& Dew [C];P4_L10_Vai.C& Abs Hum [g/m3];P4_L10_Vai.C& Vapor P [hPa]"
    
    # Removed the pressure sensors, and L10 Vai, ;P4_L10_Vai.C& Temp1 [C];P4_L10_Vai.C& Temp2 [C];P4_L10_Vai.C& Hum [%RH];P4_L10_Vai.C& Dew [C];P4_L10_Vai.C& Abs Hum [g/m3];P4_L10_Vai.C& Vapor P [hPa]
    # aswell as temperature sensors , ;P4_U.LnsiC& Temp [C];P4_U.LnsiC& Hum [%RH];P4_U.LnsiC& Pres [mBar];P4_S.Ln_C&% Temp [C];P4_S.Ln_C&% Hum [%RH];P4_S.Ln_C&% Pres [mBar]
                                    # O3_U.It_C&% Temp [C];O3_U.It_C&% Hum [%RH];O3_U.It_C&% Pres [mBar];
                                    # P4_S.It_C&% Temp [C];P4_S.It_C&% Hum [%RH];P4_S.It_C&% Pres [mBar];

     #Tid;
     # ;P4_S.It_C&% Temp [C]; P4_S.It_C&% Hum [%RH]; P4_S.It_C&% Pres [mBar];
     # ;P4_SI_T;P4_SI_H;P4_SI_P
    # P4_L10_C&% Temp [C];P4_L10_C&% Hum [%RH]; P4_L10_C&% Pres [mBar];
    # P4_L9_C&% Temp [C];P4_L9_C&% Hum [%RH];P4_L9_C&% Pres [mBar];
    # P4_L6_C&% Temp [C];P4_L6_C&% Hum [%RH];P4_L6_C&% Pres [mBar];
    # P4_U.LnsiC& Temp [C];P4_U.LnsiC& Hum [%RH];P4_U.LnsiC& Pres [mBar];
    # P4_UL_T, P4_UL_H, P4_UL_P
    # ;P4_S.Ln_C&% Temp [C];P4_S.Ln_C&% Hum [%RH];P4_S.Ln_C&% Pres [mBar]
    # ;P4_SL_T;P4_SL_H;P4_SL_P
    # P4_L4_C&% Temp [C];P4_L4_C&% Hum [%RH];P4_L4_C&% Pres [mBar]
    # O3_U.It_C&% Temp [C];O3_U.It_C&% Hum [%RH];O3_U.It_C&% Pres [mBar];
    # O3_UI_T, O3_UI_H, O3_UI_P
    # P3_L13_C&% Temp [C];P3_L13_C&% Hum [%RH];P3_L13_C&% Pres [mBar];
    # P3_L14_C&% Temp [C];P3_L14_C&% Hum [%RH];P3_L14_C&% Pres [mBar];
    # P4_L8_C&% Temp [C];P4_L8_C&% Hum [%RH];P4_L8_C&% Pres [mBar];
    # P4_L5_C&% Temp [C];P4_L5_C&% Hum [%RH];P4_L5_C&% Pres [mBar];
    # ;P4_L7_S/YP.Pa Pres [Pa];P4_Sis/Ulk.Pa Pres [Pa];P4_L2_S/YP.Pa Pres [Pa];
    # P4_L7_S/L_P;P4_S/U_P;P4_L2_S/L_P
    # ;P3_L13_S/YP.Pa Pres [Pa];
    # ;P3_L13_S/L
    # P4_L2_S/YP.Pa Pres [Pa];
    #;P4_L2_S/L_P
    # P4_L12_Vai.C& Temp [C];P4_L12_Vai.C& Temp [C];P4_L12_Vai.C& Hum [%RH];P4_L12_Vai.C& Dew [C];P4_L12_Vai.C& Abs Hum [g/m3];P4_L12_Vai.C& Vapor P [hPa];
    # P4_L1_Vai.C&% Temp1 [C];P4_L1_Vai.C&% Temp2 [C];P4_L1_Vai.C&% Hum [%RH];P4_L1_Vai.C&% Dew [C];P4_L1_Vai.C&% Abs Hum [g/m3];P4_L1_Vai.C&% Vapor P [hPa];
    # P3_L13_Vai.C& Temp [C];P3_L13_Vai.C& Temp [C];P3_L13_Vai.C& Hum [%RH];P3_L13_Vai.C& Dew [C];P3_L13_Vai.C& Abs Hum [g/m3];P3_L13_Vai.C& Vapor P [hPa];
    # ;P4_L10_Vai.C& Temp1 [C];P4_L10_Vai.C& Temp2 [C];P4_L10_Vai.C& Hum [%RH];P4_L10_Vai.C& Dew [C];P4_L10_Vai.C& Abs Hum [g/m3];P4_L10_Vai.C& Vapor P [hPa]
    # ;P4_M10_T1;P4_M10_T2;P4_M10_H;P4_M10_D;P4_M10_AH;P4_M10_VP
    # Old columns df = df[['Tid','P4_L5_C&% Temp [C]','P4_L5_C&% Hum [%RH]','P4_L1_Vai.C&% Temp1 [C]','P4_L1_Vai.C&% Temp2 [C]','P4_L1_Vai.C&% Hum [%RH]','P4_L1_Vai.C&% Dew [C]']]
    