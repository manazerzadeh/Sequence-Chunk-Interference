import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import re
from scipy import stats
import matplotlib.cm as cm
import seaborn as sns
from typing import List
import pingouin as pg

import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import AnovaRM


from natsort import index_natsorted

path = "./CI1/CI1ChunkInterference"

path_misc = './CI1_miscs/'

g_sequences = {}
g_chunks_sizes = {}
digit_change = {}
g_sequences[0] = ['13524232514', '51423252413', '35421252143', '14325242135'] #Group 1 sequences
g_sequences[1] = ['13524232514', '51423252413', '41325242351', '14325242135'] #Group 2 sequences

seq_length = len(g_sequences[0][0])

g_chunks_sizes[0] = ['2333', '3332', '2333', '3332']  #Group1 chunking structures
g_chunks_sizes[1] = ['3332', '2333', '3332', '2333']  # Group2 chunking structures

digit_change = [5, 7, 9]

fingers = ['1', '2', '3', '4', '5']

iti = 3000 
execTime = 10000 # msecs for each trial maximum
planTime = 0 # msecs for planning before movement 
hand = 2 #left or right hand

total_sub_num = 5
# total_num_blocks = 8
num_train_blocks = 5
num_test_blocks = 5
total_num_blocks = num_train_blocks + num_test_blocks
num_trials_per_train_block = 20
num_seq_changed_trials_per_test_block = 8
num_seq_unchanged_trials_per_test_block = 8
num_random_trials_per_test_block = 8
num_trials_per_test_block = num_seq_changed_trials_per_test_block + num_seq_unchanged_trials_per_test_block + num_random_trials_per_test_block






def read_dat_file(path : str):
    return pd.read_csv(path, delimiter='\t', dtype={'seq': str, 'ChunkSize': str}, usecols=lambda column: not column.startswith("Unnamed"))




def read_dat_files_subjs_list(subjs_list: List[int]):
    """
    Reads the corresponding dat files of subjects and converts them to a list of dataframes.
    """
    return [read_dat_file(path + "_" + str(sub) + ".dat") for sub in subjs_list]



def remove_error_trials(subj: pd.DataFrame) -> pd.DataFrame:
    """
    Removes error trials from the dat file of a subject
    """

    return subj[(subj['isError'] == 0) & (subj['timingError'] == 0)]


def remove_error_presses(subj_press: pd.DataFrame) -> pd.DataFrame:

    return subj_press[(subj_press['isPressError']) == 0]



def remove_next_error_presses(subj_press: pd.DataFrame) -> pd.DataFrame:

    error_incremented = subj_press[subj_press['isPressError'] == 1].copy()
    error_incremented['N'] = (error_incremented['N'] + 1)

    subj_press = subj_press.merge(error_incremented[['BN','TN', 'SubNum', 'N']], on = ['BN','TN', 'SubNum', 'N'], how= 'left', indicator=True)
    subj_press = subj_press[subj_press['_merge'] == 'left_only']
    return subj_press


def remove_remaining_next_error_presses(subj_press: pd.DataFrame) -> pd.DataFrame:
    error_rows = subj_press[subj_press['isPressError'] == 1]

    # Find the max N for each group where isPressError is 1
    max_n_for_error = error_rows.groupby(['BN','TN','SubNum'])['N'].min().reset_index()


    # Merge this information back to the original df to find the max N for each group in the original df
    press_with_max_n = subj_press.merge(max_n_for_error, on=['BN', 'TN', 'SubNum'], how='left', suffixes=('', '_max')).fillna(np.inf)

    # Filter out rows where N is more than the max N in the error rows
    press_filtered = press_with_max_n[press_with_max_n['N'] <= press_with_max_n['N_max']].drop(columns=['N_max'])

    return press_filtered


def add_IPI(subj: pd.DataFrame):
    """
    Adds interpress intervals to a subject's dataframe
    """

    for i in range(seq_length-1):
        col1 = 'pressTime'+str(i+1)
        col2 = 'pressTime'+str(i+2)
        new_col = 'IPI'+str(i+1)
        subj[new_col] = subj[col2] - subj[col1]

    # subj['IPI0'] = subj['RT']




def finger_melt_IPIs(subj: pd.DataFrame) -> pd.DataFrame:
    """
    Creates seperate row for each IPI in the whole experiment adding two columns, "IPI_Number" determining the order of IPI
    and "IPI_Value" determining the time of IPI
    """

    
    subj_melted = pd.melt(subj, 
                    id_vars=['BN', 'TN', 'SubNum', 'group', 'hand', 'isTrain', 'seq', 'ChunkSize', 'digitChangePos', 'digitChangeValue', 'isError', 'timingError'], 
                    value_vars =  [_ for _ in subj.columns if _.startswith('IPI')],
                    var_name='IPI_Number', 
                    value_name='IPI_Value')
    

    subj_melted['N'] = (subj_melted['IPI_Number'].str.extract('(\d+)').astype('int64') + 1)

    

    
    return subj_melted


def finger_melt_presses(subj: pd.DataFrame) -> pd.DataFrame:

    subj_melted = pd.melt(subj, 
                    id_vars=['BN', 'TN', 'SubNum', 'group', 'hand', 'isTrain', 'seq', 'ChunkSize', 'digitChangePos', 'digitChangeValue', 'isError', 'timingError'], 
                    value_vars =  [_ for _ in subj.columns if _.startswith('press') and not _.startswith('pressTime')],
                    var_name='Press_Number', 
                    value_name='Press_Value')
    

    subj_melted['N'] = subj_melted['Press_Number'].str.extract('(\d+)').astype('int64')

    return subj_melted


def finger_melt_responses(subj: pd.DataFrame) -> pd.DataFrame:

    subj_melted = pd.melt(subj, 
                    id_vars=['BN', 'TN', 'SubNum', 'group', 'hand', 'isTrain', 'seq', 'ChunkSize', 'digitChangePos', 'digitChangeValue', 'isError', 'timingError'], 
                    value_vars =  [_ for _ in subj.columns if _.startswith('response')],
                    var_name='Response_Number', 
                    value_name='Response_Value')
    
    subj_melted['N'] = subj_melted['Response_Number'].str.extract('(\d+)').astype('int64')

    return subj_melted


def finger_melt(subj: pd.DataFrame) -> pd.DataFrame:
    melt_IPIs = finger_melt_IPIs(subj)
    melt_presses = finger_melt_presses(subj)
    melt_responses = finger_melt_responses(subj)
    merged_df = melt_IPIs.merge(melt_presses, on = ['BN', 'TN', 'SubNum', 'group', 'hand', 'isTrain', 'seq', 'ChunkSize',
                                               'digitChangePos', 'digitChangeValue', 'isError', 'timingError', 'N'])\
                                               .merge(melt_responses, on = ['BN', 'TN', 'SubNum', 'group', 'hand', 'isTrain', 'seq', 'ChunkSize',
                                               'digitChangePos', 'digitChangeValue', 'isError', 'timingError', 'N'] )

    return add_press_error(merged_df)


def add_press_error(merged_df):
    merged_df['isPressError'] = ~(merged_df['Press_Value'] == merged_df['Response_Value'])
    return merged_df




def RT_pattern(subj: pd.DataFrame):
    """
    Plotting the average reaction times of the subject for each block
    """
    
    
    BN_grouped = subj.groupby('BN')
    plt.plot(BN_grouped['RT'].agg('median'))
    plt.xticks(range(1,11))
    plt.ylabel('RT')
    plt.xlabel('Block')
    plt.show()



def IPI_pattern_seq(subj: pd.DataFrame):
    """
    Plotting average of each IPI in each block grouped by sequence
    """
    
    BN_seq_goruped = subj.drop(columns='ChunkSize').groupby(['BN', 'seq'])
    BN_seq_agg = BN_seq_goruped.agg('median')[['RT'] + ['IPI' + str(_) for _ in range(1, seq_length)]]

    cmap = cm.Pastel1
    fig, axs = plt.subplots(num_train_blocks,1, figsize = (20,15))
    for i in range(num_train_blocks):
        block_ipis = BN_seq_agg.loc[i+1].T
        for j, seq in enumerate(block_ipis):
            axs[i].plot(block_ipis.index, block_ipis[seq], label = f'Seq {seq}', color = cmap( j), linewidth = 3)
            axs[i].legend(loc = 'upper right')
            axs[i].set_xticks(block_ipis.index)
            axs[i].set_xticklabels(block_ipis.index, rotation=45)
    plt.show()


def IPI_pattern_chunk(subj: pd.DataFrame):
    """
    Plotting average of each IPI in each block grouped by chunking pattern
    """

    cmap = cm.Pastel1
    fig, axs = plt.subplots(num_train_blocks,1, figsize = (20,15))

    for idx, (block , block_data) in enumerate(subj.drop(columns='seq').groupby(['BN'])):
        for i, (chunk, chunk_data) in enumerate(block_data.groupby('ChunkSize')):
            chunk_data_mean = chunk_data.groupby(['SubNum'])[['RT'] + ['IPI' + str(_) for _ in range(1, seq_length)]].median()
            axs[idx].errorbar(chunk_data_mean.columns, chunk_data_mean.mean(), chunk_data_mean.std(), color = cmap(i), label = chunk)
            axs[idx].legend()


    plt.show()


def between_calc(row: pd.Series):
    """
    Calculates the sum of between chunk IPIs in a trial
    """

    chunksList = [int(char) for char in row['ChunkSize']]
    cumulative_chunks_lists = [sum(chunksList[:i+1]) for i in range(len(chunksList))]
    return sum([row['IPI' + str(cumulative_chunks_lists[_])] for _ in range(len(row['ChunkSize']) - 1)])

def flag_between_within_press(row: pd.Series):
        """
        Calculates whether the fingpress is a within chunk IPI 
        """

        chunksList = [int(char) for char in row['ChunkSize']]
        cumulative_chunks_lists = ['IPI' + str(sum(chunksList[:i+1])) for i in range(len(chunksList))]
        return (row['IPI_Number'] in cumulative_chunks_lists)


def within_calc(row):
    """
    Calculates the sum of within chunk IPIs in a trial
    """
    
    return row.filter(regex = '^IPI*').sum() - between_calc(row)


def extract_sum_and_count_IPI(subj):
    """
    Adds the sum time of within and between IPIs and their count
    """
    subj['between_IPI_sum'] = subj.apply(between_calc, axis = 1)
    subj['within_IPI_sum'] = subj.apply(within_calc, axis = 1)
    subj['count_between_IPI'] = subj.apply(lambda row: len(row['ChunkSize']) - 1, axis = 1)
    subj['count_within_IPI'] = subj.apply(lambda row: seq_length - len(row['ChunkSize']), axis = 1)

def extract_train(subj):
    """
    Filters train trials and adds the sum of within and between IPIs as rows
    """

    subj_train = subj[subj['BN'] <=5]
    extract_sum_and_count_IPI(subj_train)
    return subj_train


def extract_test(subj):
    """
    Filters test trials and adds the sum of within and between IPIs as rows
    """

    subj_test = subj[subj['BN'] >5]
    extract_sum_and_count_IPI(subj_test)
    return subj_test



# def within_between_trend(subj):
    train_BN_grouped = subj.select_dtypes('number').groupby(['BN'])
    train_BN_agg = train_BN_grouped.agg('median')
    # print(train_BN_agg)
    plt.plot(range(1, num_train_blocks+ 1), train_BN_agg['between_IPI_sum']/train_BN_agg['count_between_IPI'], label = 'between IPI', marker = 'o')
    plt.plot(range(1, num_train_blocks + 1), train_BN_agg['within_IPI_sum']/train_BN_agg['count_within_IPI'], label = 'within IPI', marker = 'o')
    plt.xticks(range(1, num_train_blocks + 1))
    plt.xlabel('Block')
    plt.legend()


    _, between_p = stats.ttest_ind(train_BN_grouped.get_group(1)['between_IPI_sum']/train_BN_grouped.get_group(1)['count_between_IPI'], train_BN_grouped.get_group(5)['between_IPI_sum']/train_BN_grouped.get_group(5)['count_between_IPI'])
    _ , within_p = stats.ttest_ind(train_BN_grouped.get_group(1)['within_IPI_sum']/train_BN_grouped.get_group(1)['count_within_IPI'], train_BN_grouped.get_group(5)['within_IPI_sum']/train_BN_grouped.get_group(5)['count_within_IPI'])



def within_between_trend_train(subj_presses: pd.DataFrame):
    """
    Plot the the trend of within and between chunk IPIs during training
    """

    train_press_type_BN_grouped = subj_presses[subj_presses['isTrain'] == 1].groupby(['SubNum', 'BN', 'is_between'])

    train_press_type_BN_agg = train_press_type_BN_grouped.agg({
    'IPI_Value': 'median'
    }).reset_index()


    print(pg.rm_anova(dv = 'IPI_Value', data = train_press_type_BN_agg, within=['is_between', 'BN'], subject = 'SubNum'))


    cmap = cm.Pastel1

    for i in range(1, total_sub_num + 1):
        plt.plot(range(1, num_train_blocks+ 1), train_press_type_BN_agg[(train_press_type_BN_agg['SubNum'] == i) & (train_press_type_BN_agg['is_between'] == False)]['IPI_Value'], 
                label = 'within_IPI', color = 'red', alpha = 0.2, marker = 'o')
        plt.plot(range(1, num_train_blocks+ 1), train_press_type_BN_agg[(train_press_type_BN_agg['SubNum'] == i) & (train_press_type_BN_agg['is_between'] == True)]['IPI_Value'], 
                label = 'between_IPI', color = 'blue', alpha = 0.2, marker = 'o')

    train_press_type_BN_agg = train_press_type_BN_agg.groupby(['BN', 'is_between']).agg({
        'IPI_Value': ['mean', 'std']
    }).reset_index()

    plt.errorbar(range(1, num_train_blocks+ 1), train_press_type_BN_agg[(train_press_type_BN_agg['is_between'] == False)]['IPI_Value']['mean'], yerr = train_press_type_BN_agg[(train_press_type_BN_agg['is_between'] == False)]['IPI_Value']['std'] 
            ,label = 'within_IPI', color = 'red', alpha = 1, fmt = '-o')
    plt.errorbar(range(1, num_train_blocks+ 1), train_press_type_BN_agg[(train_press_type_BN_agg['is_between'] == True)]['IPI_Value']['mean'], yerr = train_press_type_BN_agg[(train_press_type_BN_agg['is_between'] == True)]['IPI_Value']['std'],
            label = 'between_IPI', color = 'blue', alpha = 1, fmt = '-o')


    plt.xticks(range(1, num_train_blocks + 1))
    plt.xlabel('Block')

    colors = ['blue', 'red']
    color_labels = ['within_IPI', 'between_IPI']
    color_handles = [plt.Line2D([0], [0], color=c, label=l) for c, l in zip(colors, color_labels)]
    plt.gca().add_artist(plt.legend(handles=color_handles, loc='upper right', title='Colors'))

    plt.show()



def is_digit_changed(row: pd.Series):
    """
    Determines if a digit change happened to that trial/press comparing to train
    """

    
    return int(int(row['seq'][row['digitChangePos']]) != row['digitChangeValue'])


def is_rand(row: pd.Series):
    return row['seq'] not in g_sequences[row['group']]

    


def get_test_changed_unchanged_rand(subj_test: pd. DataFrame) -> List[pd.DataFrame]:
    """
    Returns test trials/presses based on their conditions of being changed, unchanged, or random.
    """
    subj_test['isChanged'] = subj_test.apply(is_digit_changed, axis = 1)

    subj_test['isRand'] = subj_test.apply(is_rand, axis = 1)


    subj_test_seq = subj_test[subj_test['isRand'] == 0] #Rows corresponding to the original sequences either changed or unchanged

    subj_test_rand = subj_test[subj_test['isRand'] == 1] #Rows corresponding to random sequences


    subj_test_changed = subj_test_seq[subj_test_seq['isChanged'] == 1]
    subj_test_unchanged = subj_test_seq[subj_test_seq['isChanged'] == 0]

    return subj_test_changed, subj_test_unchanged, subj_test_rand


# def within_between_trend_test(subj_test):
    subj_test_changed, subj_test_unchanged, subj_test_rand = get_test_changed_unchanged_rand(subj_test)

    test_BN_changed_agg = subj_test_changed.select_dtypes('number').groupby(['BN'])
    test_BN_changed_agg = test_BN_changed_agg.agg('median')

    test_BN_unchanged_agg = subj_test_unchanged.select_dtypes('number').groupby(['BN'])
    test_BN_unchanged_agg = test_BN_unchanged_agg.agg('median')


    test_BN_rand_agg = subj_test_rand.select_dtypes('number').groupby(['BN'])
    test_BN_rand_agg = test_BN_rand_agg.agg('median')

    cmap = cm.Pastel1

    plt.plot(range(1, num_test_blocks+ 1), test_BN_changed_agg['between_IPI_sum']/test_BN_changed_agg['count_between_IPI'], label = 'between IPI Changed', marker = '*', color = cmap(0))
    plt.plot(range(1, num_test_blocks + 1), test_BN_changed_agg['within_IPI_sum']/test_BN_changed_agg['count_within_IPI'], label = 'within IPI Changed', marker = 'o', color = cmap(0))

    plt.plot(range(1, num_test_blocks+ 1), test_BN_unchanged_agg['between_IPI_sum']/test_BN_unchanged_agg['count_between_IPI'], label = 'between IPI unchanged', marker = '*', color = cmap(1))
    plt.plot(range(1, num_test_blocks + 1), test_BN_unchanged_agg['within_IPI_sum']/test_BN_unchanged_agg['count_within_IPI'], label = 'within IPI unchanged', marker = 'o', color = cmap(1))

    plt.plot(range(1, num_test_blocks + 1), test_BN_rand_agg['between_IPI_sum']/test_BN_rand_agg['count_between_IPI'], label = 'between IPI rand', marker = '*', color = cmap(2))
    plt.plot(range(1, num_test_blocks + 1), test_BN_rand_agg['within_IPI_sum']/test_BN_rand_agg['count_within_IPI'], label = 'within IPI rand', marker = 'o', color = cmap(2))

    plt.xticks(range(1, num_test_blocks + 1))
    plt.xlabel('Block')
    # plt.legend(loc =(1, 0.65))

    markers = ['*', 'o']
    marker_labels = ['Between', 'Within']
    marker_handles = [plt.Line2D([0], [0], marker=m, color='black', label=l, markersize=8) for m, l in zip(markers, marker_labels)]
    # plt.legend(handles=marker_handles, loc='upper left', title='Markers')


    colors = [cmap(0), cmap(1), cmap(2)]
    color_labels = ['Changed', 'Unchanged', 'Random']
    color_handles = [plt.Line2D([0], [0], color=c, label=l) for c, l in zip(colors, color_labels)]
    # plt.legend(handles=color_handles, loc='lower left', title='Colors')

    plt.gca().add_artist(plt.legend(handles=marker_handles, loc='upper right', title='Markers'))
    plt.gca().add_artist(plt.legend(handles=color_handles, loc=(0.5, 0.8), title='Colors'))

    plt.show()

    print("changed: ", stats.ttest_rel(test_BN_changed_agg['between_IPI_sum']/test_BN_changed_agg['count_between_IPI'], test_BN_changed_agg['within_IPI_sum']/test_BN_changed_agg['count_within_IPI']))
    print("unchanged: ", stats.ttest_rel(test_BN_unchanged_agg['between_IPI_sum']/test_BN_unchanged_agg['count_between_IPI'], test_BN_unchanged_agg['within_IPI_sum']/test_BN_unchanged_agg['count_within_IPI']))
    print("rand: ", stats.ttest_rel(test_BN_rand_agg['between_IPI_sum']/test_BN_rand_agg['count_between_IPI'], test_BN_rand_agg['within_IPI_sum']/test_BN_rand_agg['count_within_IPI']))    


def within_between_trend_press_test(subj_presses: pd.DataFrame):
    """
    Plotting trend of within and between chunk IPIs during test for the changed and unchanged conditions
    """
    press_tests = get_test_changed_unchanged_rand(subj_presses[subj_presses['isTrain'] == 0])

    cmap = cm.Pastel1
    for idx, press_test in enumerate(press_tests[:2]):
        press_test_grouped = press_test.groupby(['BN', 'SubNum', 'is_between'])

        press_test_agg = press_test_grouped.agg({
        'IPI_Value': 'median'
        }).reset_index()
        press_test_agg = press_test_agg.groupby(['BN', 'is_between']).agg({
        'IPI_Value': ['mean', 'std']
        }).reset_index()
        # plt.errorbar(range(1, num_test_blocks+ 1), press_test_agg[(press_test_agg['is_between'] == False)]['IPI_Value']['mean'], 
        # yerr = press_test_agg[(press_test_agg['is_between'] == False)]['IPI_Value']['std'] 
        #     ,label = 'within_IPI', color = cmap(idx), alpha = 1, fmt = '-o', capsize=5)
        # plt.errorbar(range(1, num_test_blocks+ 1), press_test_agg[(press_test_agg['is_between'] == True)]['IPI_Value']['mean'], 
        # yerr = press_test_agg[(press_test_agg['is_between'] == True)]['IPI_Value']['std'],
        #         label = 'between_IPI', color = cmap(idx), alpha = 1, fmt = '-^', capsize=5)
        plt.plot(range(1, num_test_blocks+ 1), press_test_agg[(press_test_agg['is_between'] == False)]['IPI_Value']['mean'],label = 'within_IPI', color = cmap(idx), alpha = 1, marker = 'o' , markersize = 10)
        plt.plot(range(1, num_test_blocks+ 1), press_test_agg[(press_test_agg['is_between'] == True)]['IPI_Value']['mean'],label = 'between_IPI', color = cmap(idx), alpha = 1, marker = '^', markersize = 10)
        

    markers = ['^', 'o']
    marker_labels = ['Between', 'Within']
    marker_handles = [plt.Line2D([0], [0], marker=m, color='black', label=l, markersize=8) for m, l in zip(markers, marker_labels)]

    colors = [cmap(0), cmap(1)]
    color_labels = ['Changed', 'Unchanged']
    color_handles = [plt.Line2D([0], [0], color=c, label=l) for c, l in zip(colors, color_labels)]

    plt.gca().add_artist(plt.legend(handles=marker_handles, loc='upper right', title='Markers'))
    plt.gca().add_artist(plt.legend(handles=color_handles, loc=(0.5, 0.8), title='Colors'))

    plt.xticks(range(1, num_test_blocks + 1))
    plt.xlabel('Block')

    plt.show()


def calculate_changed_unchanged_press_diff(subj_presses: pd.DataFrame) -> pd.DataFrame:
    press_test_changed, press_test_unchanged, _ = get_test_changed_unchanged_rand(subj_presses)

    unchanged_agg = press_test_unchanged.groupby(['BN', 'SubNum', 'IPI_Number']).agg({
        'IPI_Value': 'median'
    }).reset_index()


    changed_agg = press_test_changed.groupby(['BN', 'SubNum', 'IPI_Number', 'digitChangePos']).agg({
        'IPI_Value': 'median'
    }).reset_index()

    press_diff = changed_agg.merge(unchanged_agg, on=['BN', 'SubNum', 'IPI_Number'], suffixes=('_changed', '_unchanged'))
    press_diff['diff'] = (press_diff['IPI_Value_changed'] - press_diff['IPI_Value_unchanged'])
    press_diff['diff_rel'] = (press_diff['IPI_Value_changed'] - press_diff['IPI_Value_unchanged'])/press_diff['IPI_Value_unchanged']
    return press_diff


def finger_press_diff_trend_block(press_diff: pd.DataFrame, changed_finger: int):
    press_diff_finger = press_diff[press_diff['digitChangePos'] == changed_finger]

    cmap = cm.Pastel1

    fig, axs = plt.subplots(num_test_blocks, figsize= (20,15))
    for idx , (blocknum, block) in enumerate(press_diff_finger.groupby('BN')):
        subj_agg = block.groupby(['IPI_Number', 'SubNum']).agg({
            'diff': 'median'
        }).reset_index()
        for subnum, subj in subj_agg.groupby(['SubNum']):
            subj = subj.iloc[index_natsorted(subj['IPI_Number'])].reset_index(drop=True)
            axs[idx].plot(subj['IPI_Number'], subj['diff'], color = cmap(1), alpha = 0.2)


        block_agg = subj_agg.groupby(['IPI_Number']).agg({
            'diff' : ['mean' , 'std']
        }).reset_index()
        block_agg = block_agg.iloc[index_natsorted(block_agg['IPI_Number'])].reset_index(drop=True)
        axs[idx].errorbar(block_agg['IPI_Number'], block_agg['diff']['mean'], yerr = block_agg['diff']['std'])

        for ipi_number, ipi in subj_agg.groupby(['IPI_Number']):
            stat, p_value = stats.ttest_1samp(ipi['diff'], 0)

            if p_value < 0.05:
                axs[idx].annotate('*', (ipi_number, ipi['diff'].mean()), textcoords="offset points", xytext=(0,15), ha='center')


        axs[idx].axhline(y = 0, color = 'black', linestyle = '--')
        axs[idx].set_title(f'Block{blocknum}')



def diff_trend_test(subj_test: pd.DataFrame, changed_finger: int):
    subjs_test_changed, subjs_test_unchanged, _ = get_test_changed_unchanged_rand(subj_test)

    unchanged_agg = subjs_test_unchanged.groupby(['BN', 'SubNum', 'ChunkSize','seq']).median()[['RT'] + ['IPI' + str(_) for _ in range(1, seq_length)]].reset_index()



    changed_agg = subjs_test_changed.groupby(['BN', 'SubNum', 'ChunkSize', 'seq', 'digitChangePos']).agg('median')[['RT'] + ['IPI' + str(_) for _ in range(1, seq_length)]].reset_index()


    press_diff = changed_agg.merge(unchanged_agg, on=['BN', 'SubNum', 'ChunkSize','seq'], suffixes=('_changed', '_unchanged'))


    for i in range(1, seq_length):
        press_diff[f'IPI{i}_diff'] = press_diff[f'IPI{i}_changed'] - press_diff[f'IPI{i}_unchanged']

    press_diff['RT_diff'] = press_diff['RT_changed'] - press_diff['RT_unchanged']


    fig, axs = plt.subplots(num_test_blocks,1, figsize = (20,15))

    press_diff_finger = press_diff[press_diff['digitChangePos'] == changed_finger]
    data = {}

    for idx, (block , block_data) in enumerate(press_diff_finger.drop(columns='seq').groupby(['BN'])):
        for i, (chunk, chunk_data) in enumerate(block_data.groupby('ChunkSize')):
            chunk_data_mean = chunk_data.groupby(['SubNum'])[['RT_diff'] + ['IPI' + str(_)+"_diff" for _ in range(1, seq_length)]].mean()
            axs[idx].errorbar(chunk_data_mean.columns, chunk_data_mean.mean(), chunk_data_mean.std(), label = chunk)
            axs[idx].axhline(y = 0, color = 'black', linestyle = '--')
            axs[idx].legend()

            data[chunk] = chunk_data_mean
            # Perform significance tests and annotate the plot
        for col in chunk_data_mean.columns:
            # Get data for the current column for each chunk size
            data_2333 = data['2333'][col]
            data_3332 = data['3332'][col]
            
            # Perform the test
            stat, p_value = stats.ttest_ind(data_2333, data_3332, equal_var=False)
            
            # Annotate if significant
            if p_value < 0.05:
                axs[idx].annotate('*', (col, max(data_2333.mean(), data_3332.mean())), textcoords="offset points", xytext=(0,10), ha='center')

                
    plt.show()




def finger_press_diff_trend(press_diff: pd.DataFrame):
    fig, axs = plt.subplots(len(digit_change), figsize= (20,15))
    cmap = cm.Pastel1

    for idx, change in enumerate(digit_change):
        press_diff_finger = press_diff[press_diff['digitChangePos'] == change]
        subj_agg = press_diff_finger.groupby(['IPI_Number', 'SubNum']).agg({
            'diff': 'mean'
        }).reset_index()

        for subnum, subj in subj_agg.groupby(['SubNum']):
            subj = subj.iloc[index_natsorted(subj['IPI_Number'])].reset_index(drop=True)
            axs[idx].plot(subj['IPI_Number'], subj['diff'], color = cmap(1))


        diff_agg = subj_agg.groupby(['IPI_Number']).agg({
            'diff': ['mean', 'std']
        }).reset_index()


        diff_agg = diff_agg.iloc[index_natsorted(diff_agg['IPI_Number'])].reset_index(drop=True)
        axs[idx].errorbar(diff_agg['IPI_Number'], diff_agg['diff']['mean'], yerr = diff_agg['diff']['std'])

        for ipi_number, ipi in subj_agg.groupby(['IPI_Number']):
            stat, p_value = stats.ttest_1samp(ipi['diff'], 0)

            if p_value < 0.05:
                axs[idx].annotate('*', (ipi_number, ipi['diff'].mean()), textcoords="offset points", xytext=(0,15), ha='center')


        axs[idx].axhline(y = 0, color = 'black', linestyle = '--')
        axs[idx].set_title(f'changed digit = {change}')


def finger_press_diff_rel_trend(press_diff: pd.DataFrame):
    fig, axs = plt.subplots(len(digit_change), figsize= (20,15))
    cmap = cm.Pastel1

    for idx, change in enumerate(digit_change):
        press_diff_finger = press_diff[press_diff['digitChangePos'] == change]
        subj_agg = press_diff_finger.groupby(['IPI_Number', 'SubNum']).agg({
            'diff_rel': 'median'
        }).reset_index()

        for subnum, subj in subj_agg.groupby(['SubNum']):
            subj = subj.iloc[index_natsorted(subj['IPI_Number'])].reset_index(drop=True)
            axs[idx].plot(subj['IPI_Number'], subj['diff_rel'], color = cmap(1))


        diff_agg = subj_agg.groupby(['IPI_Number']).agg({
            'diff_rel': ['mean', 'std']
        }).reset_index()


        diff_agg = diff_agg.iloc[index_natsorted(diff_agg['IPI_Number'])].reset_index(drop=True)
        axs[idx].errorbar(diff_agg['IPI_Number'], diff_agg['diff_rel']['mean'], yerr = diff_agg['diff_rel']['std'])

        for ipi_number, ipi in subj_agg.groupby(['IPI_Number']):
            stat, p_value = stats.ttest_1samp(ipi['diff_rel'], 0)

            if p_value < 0.05:
                axs[idx].annotate('*', (ipi_number, ipi['diff_rel'].mean()), textcoords="offset points", xytext=(0,15), ha='center')


        axs[idx].axhline(y = 0, color = 'black', linestyle = '--')
        axs[idx].set_title(f'changed digit = {change}')
