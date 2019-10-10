import numpy as np
import pandas as pd
import yaml
from sqlalchemy import create_engine, Table, MetaData, and_, select, join, types
from sqlalchemy.sql import column
from settings import postgres_dev_str
import time

def main():
    # Read tables
    cnx = create_engine(postgres_dev_str)
    heroes = pd.read_sql_table('hero_info', cnx)
    drafts = pd.read_sql_table('drafts', cnx)
    match_info = pd.read_sql_table('match_table', cnx)

    # Get Pick lists
    drafts_picks = drafts[drafts.is_pick]
    drafts_picks = drafts_picks.merge(heroes[['id', 'name']], left_on = 'hero_id', \
        right_on = 'id')
    drafts_picks = drafts_picks.sort_values(by = ['match_id', 'order'])
    drafts_sentences = drafts_picks.groupby(['match_id', 'team'])['name'].\
        apply(list)
    drafts_sentences = drafts_sentences.reset_index()
    drafts_sentences.columns = ['match_id', 'team', 'draft']
    drafts_sentences = drafts_sentences.merge(match_info[['match_id', \
        'radiant_win', 'start_time']], left_on = 'match_id', right_on = 'match_id',\
            how = 'left')

    # Get the draft outcome
    drafts_sentences['outcome'] = np.where((np.array(drafts_sentences['team'] == 0) & \
        np.array(drafts_sentences['radiant_win'] == True)) |\
             (np.array(drafts_sentences['team'] == 1) & \
                 np.array(drafts_sentences['radiant_win'] == False)), 1, 0)

    # Get Time Weights
    current_time = int(time.time()) 
    drafts_sentences['time_elapsed'] = current_time - drafts_sentences[['start_time']]
    drafts_sentences['normalized_time'] = drafts_sentences['time_elapsed'] / \
        drafts_sentences['time_elapsed'].sum().sum()
    drafts_sentences['normalized_time'] = drafts_sentences['normalized_time'] / \
        drafts_sentences['normalized_time'].max()

    return drafts_sentences

if __name__ == '__main__':
    main()