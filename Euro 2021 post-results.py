'''Post results we simulate success of the algorithm'''

import requests
from bs4 import BeautifulSoup
import sys
import numpy as np
import pandas as pd
import itertools
import random
from timeit import default_timer as timer

# Time decorator to find easy inefficiencies
def time(func):
    def wrapper(*arg, **kwargs):
        start = timer()
        result = func(*arg, **kwargs)
        end = timer()
        print(f'{func.__name__} ran in {end-start:.06f}')
        return result
    return wrapper


# Global declaration of groups
groups = {'A': ['Italy', 'Switzerland', 'Turkey', 'Wales'],
          'B': ['Belgium', 'Denmark', 'Finland', 'Russia'],
          'C': ['Austria', 'Netherlands', 'North Macedonia', 'Ukraine'],
          'D': ['Croatia', 'Czech Republic', 'England', 'Scotland'],
          'E': ['Poland', 'Slovakia', 'Spain', 'Sweden'],
          'F': ['France', 'Germany', 'Hungary', 'Portugal']}


def scrape_odds_from_web(get_new_data):
    '''Scrape the latest odds from oddschecker.com for winner of Euro 2020
    Return dataframe of probability of winning the tournament the bookies are offering'''

    html_local_name = 'oddschecker_euro_2020_odds.html'
    if get_new_data:
        url = 'https://www.oddschecker.com/football/euro-2020/winner'

        # Need a header from browser to not be blocked from scraping
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.77 Safari/537.36'}

        r = requests.get(url, headers=headers)
        html = r.content

        # Save the html for future reference
        with open(html_local_name, 'wb') as f:
            f.write(html)

    else:  # read from saved data
        with open(html_local_name, 'rb') as f:
            html = f.read()

    # Parse the html and extract odds
    soup = BeautifulSoup(html, 'html.parser')

    # Find odds by tr
    containers = soup.find_all('tr', {'class': 'diff-row evTabRow bc'})
    probs = {}
    for container in containers:
        # Odds as text in each td
        td = container.find_all('td')
        # drop empty
        country, *odds = [val.text for val in td if val.text != '']

        # Adjust odds to float and evalute
        odds = list(map(eval, odds))

        # Adjust to probability
        prob = [1/(odd+1) for odd in odds]

        probs[country] = prob

    # Create df of odds for usability
    df_prob = pd.DataFrame(columns=['Country', 'Min', 'Max', 'Mean', 'Median'])
    for country in probs:
        df_prob = df_prob.append({'Country': country, 'Min': min(probs[country]), 'Max': max(probs[country]),
                                  'Mean': np.mean(probs[country]), 'Median': np.median(probs[country])},
                                 ignore_index=True)

    return df_prob


# Lambda for median val probability from country name
def m(x): return df_prob.loc[df_prob['Country'] == x]['Median'].iloc[0]


def gen_result(p1, p2, u=2.88):
    '''Take in prob team 1 winning and team 2 and generate a score
    We generate with poisson distribution where u = 2.88 goals per game
    2.88 is taken from average of English leagues games from 1888-2016: http://rstudio-pubs-static.s3.amazonaws.com/337949_a6b294c25d75426eaf0b6bbee8b55175.html
    Returns: p1 goals, p2 goals and winner if draw
    
    TODO: Perhaps can improve goal predictions somewhat, these didn't do very well. Predicting 4-0 etc. is unlikely?'''

    # Ratio of p1 wins vs p2 for future calcs
    # Initially took a simple ratio but since individual games are more stochastic than final winner I wanted to normalise a bit
    # Square root seemed a bit too far so settled on the golden ratio root for good fun
    phi = (1 + 5**0.5) / 2  # golden ratio
    def rt(x): return x**(1/phi)
    ratio = rt(p1)/(rt(p1)+rt(p2))

    # Poission u parameters
    p1_u_goals = ratio * u  # mean goals for team 1
    p2_u_goals = u - p1_u_goals

    # Generate goals
    p1_goals = np.random.poisson(lam=p1_u_goals)
    p2_goals = np.random.poisson(lam=p2_u_goals)

    # Generate a winner if needed
    if p1_goals > p2_goals:
        winner = 1
    elif p2_goals > p1_goals:
        winner = 2
    else:
        # Generate a winner if a draw by betting odds
        if ratio > np.random.rand():
            winner = 1
        else:
            winner = 2

    # Return score and winner
    return p1_goals, p2_goals, winner


def get_df_group_matches():
    '''Return df of groups for Euro 2020'''

    cols = ['Team1', 'Team2', 'Team1_Goals', 'Team2_Goals']
    df_group_matches = pd.DataFrame(columns=cols)

    for group in groups:
        combos = itertools.combinations(groups[group], 2)
        for c in combos:
            *results, winner = gen_result(m(c[0]), m(c[1]))
            df_group_matches = df_group_matches.append(dict(zip(cols, [c[0], c[1]] + list(results))),
                                                       ignore_index=True)

    return df_group_matches


def standings_from_results(matches):
    '''Take results of group matches and return df of group standings'''
    # Generate group standings from results
    df_group = pd.DataFrame(groups.items(), columns=[
                            'Group', 'Country']).explode(column='Country')

    # Keep only countries in matches
    df_group = df_group.loc[df_group['Country'].isin(
        matches['Team1']) | df_group['Country'].isin(matches['Team2'])]

    # Country as index
    df_group = df_group.set_index('Country')

    # Add data columns
    for d_col in ['W', 'D', 'L', 'GF', 'GA', 'GD', 'Pts']:
        df_group[d_col] = 0

    # For each match adjust the board
    for index, match in matches.iterrows():
        # W, D, L and Pnts
        if match['Team1_Goals'] > match['Team2_Goals']:
            df_group.loc[match['Team1'], 'W'] += 1
            df_group.loc[match['Team1'], 'Pts'] += 3
            df_group.loc[match['Team2'], 'L'] += 1
        elif match['Team1_Goals'] < match['Team2_Goals']:
            df_group.loc[match['Team2'], 'W'] += 1
            df_group.loc[match['Team2'], 'Pts'] += 3
            df_group.loc[match['Team1'], 'L'] += 1
        else:
            df_group.loc[match['Team1'], 'D'] += 1
            df_group.loc[match['Team1'], 'Pts'] += 1
            df_group.loc[match['Team2'], 'D'] += 1
            df_group.loc[match['Team2'], 'Pts'] += 1

        # GF (goals for) and GA (goals against)
        df_group.loc[match['Team1'], 'GF'] += match['Team1_Goals']
        df_group.loc[match['Team2'], 'GF'] += match['Team2_Goals']

        df_group.loc[match['Team1'], 'GA'] += match['Team2_Goals']
        df_group.loc[match['Team2'], 'GA'] += match['Team1_Goals']

    # GD
    df_group['GD'] = df_group['GF'] - df_group['GA']

    return df_group


def get_group_rankings(df_group):
    '''Determine group rankings from the results'''
    df_group_results = df_group.copy()
    df_group_results['Rank'] = df_group_results.groupby(
        'Group')['Pts'].rank(method='min', ascending=False)

    # To determine ties, most points in head to head (e.g. if 3 not just w/l) goal difference, goals scored
    # For each group if rank tie then check just those matches
    for group in df_group_results['Group'].unique():
        df_slice_1 = df_group_results.loc[df_group_results['Group'] == group]
        for rank in df_slice_1['Rank'].unique():
            df_slice_2 = df_slice_1.loc[df_slice_1['Rank'] == rank]
            if len(df_slice_2) > 1:  # Then we have some ties to determine
                # First keep only matches between these teams and rank again
                countries = df_slice_2.index
                filter1 = df_group_matches['Team1'].isin(countries)
                filter2 = df_group_matches['Team2'].isin(countries)
                matches = df_group_matches.loc[filter1 & filter2]

                # Then generate results from just those matches
                standings = standings_from_results(matches)

                # Join pts, gd, gf back to slice for ranking 2
                standings = standings[['Pts', 'GD', 'GF']]
                standings.columns = ['Pts2', 'GD2', 'GF2']
                if 'Pts2' in df_group_results.columns:
                    df_group_results.update(standings)
                else:
                    df_group_results = df_group_results.merge(
                        standings, on='Country', how='left')

                # Now Rank2 by: Rank >Pts2 > GD2 > GF2 > GD > GF > W
                # https://www.mirror.co.uk/sport/football/news/euro-2020-group-stages-points-24250173
                rank_order = ['Rank', 'Pts2', 'GD2', 'GF2', 'GD', 'GF', 'W']
                rank_asc_desc = [True, False, False,
                                 False, False, False, False]
                df_group_results = df_group_results.sort_values(
                    rank_order, ascending=rank_asc_desc)
                df_group_results['TempRank'] = [
                    i for i in range(len(df_group_results))]
                df_group_results['Rank2'] = df_group_results.groupby('Group')[
                    'TempRank'].rank()

    # Re-order
    # If we didn't need a rank2 col then rename for future convenience
    if 'Rank2' not in df_group_results.columns:
        df_group_results = df_group_results.rename(columns={'Rank': 'Rank2'})
    df_group_results = df_group_results.sort_values(['Group', 'Rank2'])

    # Drop cols
    if 'TempRank' in df_group_results.columns:
        df_group_results = df_group_results.drop(columns=['TempRank'])

    return df_group_results


def get_third_place_rankings(df_group_results):
    '''Four 3rd place teams to go through. Ranked by Pts > GD > GF
    https://www.thesun.co.uk/sport/football/15062810/euro-2020-third-place-group-qualify-knockout-stages/'''

    df_3rd = df_group_results.loc[df_group_results['Rank2'] == 3].copy()

    rank_order = ['Pts', 'GD', 'GF']

    df_3rd['Rank3'] = df_3rd[rank_order].apply(
        tuple, axis=1).rank(ascending=False)

    return df_3rd


def get_knockout_pairing(df_group_results, df_3rd):
    '''Knockout round pairing (ro16) from group results
    https://en.wikipedia.org/wiki/UEFA_Euro_2020_knockout_phase#Combinations_of_matches_in_the_round_of_16'''

    # 3rd place matching. 1B, 1C, 1E, 1F vs the qualifiers of:
    # Create generator for match order
    first_v_third = [
        ['A', 'D', 'B', 'C'],
        ['A', 'E', 'B', 'C'],
        ['A', 'F', 'B', 'C'],
        ['D', 'E', 'A', 'B'],
        ['D', 'F', 'A', 'B'],
        ['E', 'F', 'B', 'A'],
        ['E', 'D', 'C', 'A'],
        ['F', 'D', 'C', 'A'],
        ['E', 'F', 'C', 'A'],
        ['E', 'F', 'D', 'A'],
        ['E', 'D', 'B', 'C'],
        ['F', 'D', 'C', 'B'],
        ['F', 'E', 'C', 'B'],
        ['F', 'E', 'D', 'B'],
        ['F', 'E', 'D', 'C']
    ]

    # Take top 4 teams with nsmallest so if dupe we coinflip
    keep = random.choice(['first', 'last'])
    qual_3rd = df_3rd.nsmallest(4, 'Rank3', keep=keep)['Group'].tolist()
    #qual_3rd = df_3rd.loc[df_3rd['Rank3'] <= 4]['Group'].tolist()
    third_pairing = [lst for lst in first_v_third if sorted(lst) == sorted(qual_3rd)][0]

    def gen_third_place():
        yield third_pairing[0]
        yield third_pairing[3]
        yield third_pairing[2]
        yield third_pairing[1]

    third_place = gen_third_place()

    # And a standard method to fetch the team from the df using our notation
    def get_team(group_rank):
        '''Method to fetch results for knockout phase pairing'''
        if group_rank == '3':
            return df_3rd[df_3rd['Group'] == next(third_place)].index[0]
        else:
            group, rank = list(group_rank)
            return df_group_results[(df_group_results['Group'] == group) & (df_group_results['Rank2'] == int(rank))].index[0]

    # Define knockout round grouping starting with round of 16
    round_16_pairs = [
        ['B1', '3'],
        ['A1', 'C2'],
        ['F1', '3'],
        ['D2', 'E2'],
        ['E1', '3'],
        ['D1', 'F2'],
        ['C1', '3'],
        ['A2', 'B2']
    ]

    df_ko = pd.DataFrame(round_16_pairs, columns=['Team1', 'Team2'])
    df_ko['Team1'] = df_ko['Team1'].apply(get_team)
    df_ko['Team2'] = df_ko['Team2'].apply(get_team)

    return df_ko


def gen_knockout_results(df_ko):
    '''Pass in knockout df, generate match results and return next round'''
    # R16, find winners, r8 -> r4 -> r2 -> winner

    # Generator
    while True:
        # Predict winner
        df_ko[['Team1_Goals', 'Team2_Goals', 'Winner']] = df_ko[['Team1', 'Team2']].apply(
            lambda x: gen_result(m(x[0]), m(x[1])),
            axis=1, result_type='expand')

        # Yield results
        yield df_ko

        # Now reduce down to next matches
        df_ko = pd.DataFrame(
            [[df_ko.loc[i, f"Team{df_ko.loc[i, 'Winner']}"],
              df_ko.loc[i+1, f"Team{df_ko.loc[i+1, 'Winner']}"]]
             for i in range(0, len(df_ko), 2)],
            columns=['Team1', 'Team2']
        )


def add_wld_to_group_matches(df):
    '''Take in group matches and add extra col for w/l/d'''
    df['Result'] = np.where(df['Team1_Goals'] > df['Team2_Goals'], 'W',
                    np.where(df['Team1_Goals'] < df['Team2_Goals'], 'L',
                    'D'))
    return df


def compute_points(df_group_matches, df_group_results, df_ro16, df_qf_results, df_sf_results, df_f_results):
    '''Take in model predictions and compare to actual results'''
    points = 0

    # ~~~ Group stages: 1 point for correct outcome, 1 point for correct score ~~~
    # Read in actual results
    df_group_matches_act = pd.read_excel('euro_2021_results.xlsx', sheet_name='Group Matches', usecols='B:E')

    # Add win/loss/draw for comparison
    df_group_matches = add_wld_to_group_matches(df_group_matches)
    df_group_matches_act = add_wld_to_group_matches(df_group_matches_act)

    # Merge
    df_group_matches = df_group_matches.merge(df_group_matches_act, left_index=True, right_index=True)

    # 1 point per matching result, 1 point per matching score
    points_group_results = len(df_group_matches[df_group_matches['Result_x']==df_group_matches['Result_y']])
    points_group_scores = len(df_group_matches[(df_group_matches['Team1_Goals_x']==df_group_matches['Team1_Goals_y']) &
                             (df_group_matches['Team2_Goals_x']==df_group_matches['Team2_Goals_y'])])
    points += points_group_results + points_group_scores

    # ~~~ 2 points per correct group qualifier, 1 additional point for correct group winner ~~~
    # Check top of the 6 groups
    my_group_winners = set(df_group_results[df_group_results['Rank2']==1].index)
    actual_winners = {'Italy', 'Belgium', 'Netherlands', 'England', 'Sweden', 'France'}
    points_group_winners = len(actual_winners & my_group_winners)

    # And 2 points per qualifier from df_ro16 teams
    my_qualifiers = set(df_ro16['Team1']) | set(df_ro16['Team2'])
    actual_qualifiers = actual_winners | \
                        {'Portugal', 'Austria', 'Switzerland', 'Croatia', 'Spain', 'Ukraine', 'Germany', 'Czech Republic', 'Wales', 'Denmark'}
    points_qualifiers = len(actual_qualifiers & my_qualifiers) * 2

    points += points_group_winners + points_qualifiers

    # ~~~ 3 points for each of 8 quarterfinalists ~~~
    my_qf = set(df_qf_results['Team1']) | set(df_qf_results['Team2'])
    actual_qf = {'Belgium', 'Italy', 'Switzerland', 'Spain', 'Ukraine', 'England', 'Czech Republic', 'Denmark'}
    points_qf = len(actual_qf & my_qf) * 3

    points += points_qf

    # ~~~ 4 points for each of 4 semifinalists ~~~
    my_sf = set(df_sf_results['Team1']) | set(df_sf_results['Team2'])
    actual_sf = {'Italy', 'Spain', 'England', 'Denmark'}
    points_sf = len(actual_sf & my_sf) * 4

    points += points_sf


    # ~~~ 5 points for each of 2 finalists ~~~
    my_f = set(df_f_results['Team1']) | set(df_f_results['Team2'])
    actual_f = {'Italy', 'England'}
    points_f = len(actual_f & my_f) * 5

    points += points_f

    # ~~~ 10 points for the winner ~~~
    points_winner = 0
    if df_f_results.loc[0, f"Team{df_f_results.loc[0, 'Winner']}"] == 'Italy':
        points_winner = 10
    points += points_winner

    # Return result
    all_points = [points_group_results, points_group_scores, points_qualifiers, points_group_winners,
                  points_qf, points_sf, points_f, points_winner]
    assert sum(all_points) == points
    return points, all_points


def save_results_to_excel(*dfs):
    '''Take in all the dfs and save to Excel'''

    sheet_names = ['P(Win)', 'Group Matches', 'Group Standings', 'Group Results', '3rd Place Results', 'Ro16', 'QF', 'SF', 'F']

    writer = pd.ExcelWriter('euro_2021_poisson_forecast.xlsx', engine='xlsxwriter')

    for df, sheet_name in zip(dfs, sheet_names):
        df.to_excel(writer, sheet_name=sheet_name)

    writer.save()


if __name__ == '__main__':
    # Get probability of each team winning Euro according to bookies
    df_prob = scrape_odds_from_web(get_new_data=False)

    df_pnts = pd.DataFrame(columns=['Grp1', 'Grp2', 'Grp3', 'Grp4', 'KO1', 'KO2', 'KO3', 'KO4', 'Total'])
    for i in range(100):
        # Get group matches
        df_group_matches = get_df_group_matches()

        # Generate group results from the matches
        df_group = standings_from_results(matches=df_group_matches)

        # Determine rankings of groups from the results (complex)
        df_group_results = get_group_rankings(df_group)

        # Determine the top four 3rd place teams
        df_3rd = get_third_place_rankings(df_group_results)

        # Get the knockout round (ro16) match pairs
        df_ro16 = get_knockout_pairing(df_group_results, df_3rd)

        # Get ro16 - generator for knockout rounds
        knockout_results = gen_knockout_results(df_ro16)
        df_ro16_results = next(knockout_results)

        # Get quarters
        df_qf_results = next(knockout_results)

        # Get semis
        df_sf_results = next(knockout_results)

        # Get finals
        df_f_results = next(knockout_results)

        # Compute points
        # NOTE: TEMP read in my original predictions
        if False:  # NOTE: think dad got one wrong and I am 74 pnts welp
            excel = 'my results\\euro_2021_poisson_forecast.xlsx'
            df_group_matches = pd.read_excel(excel, sheet_name='Group Matches', usecols='B:E')
            df_group_results = pd.read_excel(excel, sheet_name='Group Results', index_col=0)
            df_ro16 = pd.read_excel(excel, sheet_name='Ro16', usecols='B:F')
            df_qf_results = pd.read_excel(excel, sheet_name='QF', usecols='B:F')
            df_sf_results = pd.read_excel(excel, sheet_name='SF', usecols='B:F')
            df_f_results = pd.read_excel(excel, sheet_name='F', usecols='B:F')

        points, all_points = compute_points(df_group_matches, df_group_results, df_ro16, df_qf_results, df_sf_results, df_f_results)
        df_pnts.loc[i] = all_points + [points]

    # Save our points to Excel
    df_pnts.to_excel('point_distribution.xlsx')
    print(df_pnts['Total'].mean())
    print(df_pnts['Total'].std())

    print('Done!')
