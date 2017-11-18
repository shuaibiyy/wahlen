# coding=utf-8
# Written using python 3.6
import csv
import pprint
import pandas as pd
import matplotlib.pyplot as plt
from functools import reduce

plt.rcdefaults()

NOT_PARTIES = ['Wahlberechtigte', 'Wähler', 'Ungültige', 'Gültige']


def fetch_rows(csv_path):
    """Open a CSV file and read its lines."""
    rows = []

    with open(csv_path, 'rt', encoding='utf8') as f:
        reader = csv.reader(f)
        for row in reader:
            rows.append(row)

    return rows


def get_csv_values():
    """Return the semi-colon separated values in the csv as a list."""
    rows = fetch_rows('./ergebnisse.csv')

    # Remove column headings
    del rows[0]

    return list(map(lambda x: x[0].split(';'), rows))


def values_at(csv_values, index):
    """Return all values at an index in the csv."""

    # remove entries that are not for parties.
    filtered = list(filter(lambda x: x[2] not in NOT_PARTIES, csv_values))

    return list(map(lambda x: x[index], filtered))


def unique_values_at(csv_values, index):
    """Return all unique values at an index in the csv."""
    return set(values_at(csv_values, index))


def values_by(csv_values, filter_text, index):
    """Return values filtered by the text at an index."""
    return list(filter(lambda x: x[index] == filter_text, csv_values))


def party_second_vote(party, value):
    """Return a zweitstimmen if it matches a party."""
    if value[2] == party:
        # zweitstimmen is at index #4.
        return value[4]

    return '-'


def party_second_votes(party, values):
    """Return all zweitstimmen for a party."""
    return list(map(lambda x: party_second_vote(party, x), values))


def filter_dashes(kv):
    """Remove dashes from seconds in a 2-tuple."""
    return kv[0], list(filter(lambda x: x != '-', kv[1]))


def aggregate(kv):
    """Aggregate values of all seconds in a 2-tuple."""
    return kv[0], reduce(lambda acc, x: acc + int(x), kv[1], 0)


def total(values):
    """Sum up values of all seconds in a list of 2-tuples."""
    return reduce(lambda acc, kv: acc + kv[1], values, 0)


def lookup_alt_names(alternate_names, party):
    """Look up a party's alternate names."""
    maybe = list(filter(lambda x: x[0].upper() == party[0].upper(), alternate_names))

    if not maybe:
        return []

    return maybe[0][1]


def lookup_party_votes(parties_votes, party_name):
    """Look up a party's votes."""
    maybe = list(filter(lambda x: x[0].upper() == party_name.upper(), parties_votes))

    if not maybe:
        return party_name, 0

    return maybe[0]


def merge_alt_names(alternate_names, alternates_with_votes, party):
    """Merge the votes of a party with votes of its known alternate names."""
    matched_alt_names = lookup_alt_names(alternate_names, party)
    matched_votes = list(map(lambda x: lookup_party_votes(alternates_with_votes, x), matched_alt_names))
    matched_votes.append(party)

    return party[0], total(matched_votes)


def merge_parties_alt_names(parties_votes):
    """Merge the votes of parties with votes of their known alternate names."""
    alternate_names = [
        ['DIE LINKE', ['DIE LINKE.']],
        ['GRÜNE', ['GRÜNE/B 90', 'EB: Gruner']]
    ]

    all_alternate_names = [x.upper() for x in reduce(lambda acc, y: acc + y[1], alternate_names, [])]
    alts = [f for f in parties_votes if f[0].upper() in all_alternate_names]
    originals = [f for f in parties_votes if f[0].upper() not in all_alternate_names]

    return map(lambda x: merge_alt_names(alternate_names, alts, x), originals)


def cleanse_second_votes(dirty):
    """Return a sorted list of merged parties with non-zero votes."""
    unsorted_votes = list(map(aggregate, map(filter_dashes, dirty)))
    non_zero_votes = list(filter(lambda x: x[1] != 0, unsorted_votes))
    merged_votes = merge_parties_alt_names(non_zero_votes)

    return sorted(merged_votes, key=lambda tup: tup[1], reverse=True)


def percentage(num, denom):
    """Return the percentage string of a numerator to its denominator."""
    return '{:.4f}'.format((float(num) / denom) * 100)


def votes_with_percentages(votes_with_total):
    """Return a list of parties along with their votes and percentage share."""
    return list(map(lambda kv: (kv[0], percentage(kv[1], votes_with_total[1])), votes_with_total[0]))


def second_votes():
    """Compute zweitstimmen for all parties in the csv."""
    vs = get_csv_values()
    # party names are at the 3rd index.
    parties = unique_values_at(vs, 2)

    # contains `-` values for missing votes.
    unfiltered_values = list(map(lambda x: (x, party_second_votes(x, vs)), parties))

    cleansed_votes = cleanse_second_votes(unfiltered_values)
    vote_total = total(cleansed_votes)

    return cleansed_votes, vote_total


def constituency_votes(constituency, values):
    """Return the votes for the parties in a constituency."""
    const_vals = values_by(values, constituency, 1)

    return list(map(lambda x: (x[2], x[3]), const_vals))


def constituencies_votes(state, values):
    """Return the votes for the constituencies in a state."""
    state_vals = values_by(values, state, 0)
    constituencies = unique_values_at(state_vals, 1)

    return list(map(lambda x: (x, constituency_votes(x, state_vals)), constituencies))


def cleanse_first_votes(values):
    real_parties = list(filter(lambda x: x[2] not in NOT_PARTIES, values))
    with_votes = list(filter(lambda x: x[3] != '-', real_parties))
    states = unique_values_at(values, 0)

    # NOTE: cleansing should occur (including merge) before transformations.

    # A list of states, where each state is a tuple of the state id and a list of constituents.
    # Each constituent is a tuple of the constituent id and a list of tuples of parties and their votes.
    # E.G.:
    # [[('1',
    #    [('11',
    #      [('CDU', 5000), ('DIE LINKE', 4000)]),
    #     ('15',
    #      [('DIE LINKE', 6000), ('CDU', 3435)])])]]

    states_votes = list(map(lambda x: (x, constituencies_votes(x, with_votes)), states))

    return states_votes


def first_votes():
    """Return parties in all states and constituencies with their 1st and 2nd votes."""
    vs = get_csv_values()

    cleansed = cleanse_first_votes(vs)
    return cleansed


def direktmandat_winners():
    """Return the winners of the direktmandat for each constituency.
    E.g. [['1', [('11', 'CDU'), ('4', 'DIE LINKE')], ['2', [('1', 'CDU')]]]]"""
    # For each state, get the constituencies.
    # For each constituency, get the parties and their votes, reduce to party with largest votes.


def total_below(vs, percent):
    """Return total votes below the provided percentage."""
    other_votes = list(filter(lambda x: float(x[1]) < percent, vs))
    return reduce(lambda acc, kv: acc + float(kv[1]), other_votes, 0)


def chart():
    """Display a bar chart of parties to the percentages of their votes."""
    vs = votes_with_percentages(second_votes())

    above_five_percent = list(filter(lambda x: float(x[1]) >= 5, vs))
    total_below_5_percent = total_below(vs, 5)
    above_five_percent.extend([('others', total_below_5_percent)])

    percents = list(map(lambda x: float(x[1]), above_five_percent))
    party_titles = list(map(lambda x: x[0], above_five_percent))
    x_pos = range(len(party_titles))

    colors = 'rgbkymc'

    plt.bar(x_pos, percents, align='center', color=colors, alpha=0.6)
    plt.xticks(x_pos, party_titles)
    plt.ylabel('Percentages of Votes Won')
    plt.xlabel('Parties')
    plt.title('2017 German Election Results')

    plt.show()


def chart_with_labels():
    """Display a bar chart with percentage labels for each bar."""
    vs = votes_with_percentages(second_votes())

    above_five_percent = list(filter(lambda x: float(x[1]) >= 5, vs))
    total_below_5_percent = total_below(vs, 5)
    above_five_percent.extend([('others', total_below_5_percent)])

    percents = list(map(lambda x: float(x[1]), above_five_percent))
    party_titles = list(map(lambda x: x[0], above_five_percent))

    freq_series = pd.Series.from_array(percents)

    plt.figure(figsize=(12, 8))
    ax = freq_series.plot(kind='bar')
    ax.set_title('2017 German Election Results')
    ax.set_xlabel('Parties')
    ax.set_ylabel('Percentages of Votes Won')
    ax.set_xticklabels(party_titles)

    bars = ax.patches

    labels = ['{:.2f}'.format(percents[i]) + '%' for i in range(len(percents))]

    for rect, label in zip(bars, labels):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, height, label, ha='center', va='bottom')

    plt.show()


def display_votes():
    """Display semicolon separated values of parties and their vote percentages."""
    print('Party;Votes')
    for row in second_votes()[0]:
        print('{0};{1}'.format(row[0], row[1]))


def compute_seats(dividend, total_shares, total_seats, parties_votes):
    """Seat allocation algorithm."""
    trial_seats = list(map(lambda x: (x[0], round(float(x[1]) / dividend)), parties_votes))
    total_trial_seats = total(trial_seats)

    if total_trial_seats != total_seats:
        return compute_seats(dividend - 1, total_shares, total_seats, parties_votes)

    return trial_seats, total_trial_seats


def second_vote_seats():
    """Allocate seats based on zweitstimmen."""
    total_seats = 599
    vv, shares = second_votes()
    starting_dividend = round(float(shares) / total_seats)

    allocations, allocated = compute_seats(starting_dividend, shares, total_seats, vv)
    print('Total seats allocated: ', allocated)
    print('Party;Seats')
    for row in allocations:
        print('{0};{1}'.format(row[0], row[1]))


pprint.pprint(first_votes())
