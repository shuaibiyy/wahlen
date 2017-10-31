# coding=utf-8
# Written using python 3.6
import csv
import pandas as pd
from functools import reduce
import matplotlib.pyplot as plt

plt.rcdefaults()


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


def parties():
    """Return a set of all parties in the csv."""
    vs = get_csv_values()

    # party names are at the 3rd index.
    initial = list(set(map(lambda x: x[2], vs)))

    not_parties = ['Wahlberechtigte', 'Wähler', 'Ungültige', 'Gültige']

    return list(filter(lambda x: x not in not_parties, initial))


def second_vote(party, value):
    """Return a zweitstimmen if it matches a party."""
    if value[2] == party:
        # zweitstimmen is at index #4.
        return value[4]

    return '-'


def second_votes(party, values):
    """Return all zweitstimmen for a party."""
    return list(map(lambda x: second_vote(party, x), values))


def filter_dashes(kv):
    """Remove dashes from seconds in a 2-tuple."""
    return kv[0], list(filter(lambda x: x != '-', kv[1]))


def aggregate(kv):
    """Aggregate values of all seconds in a 2-tuple."""
    return kv[0], reduce(lambda acc, x: acc + int(x), kv[1], 0)


def cleanse_votes(dirty):
    """Return a sorted list of parties with non-zero votes."""
    unsorted_votes = list(map(aggregate, map(filter_dashes, dirty)))
    non_zero_votes = list(filter(lambda x: x[1] != 0, unsorted_votes))
    return sorted(non_zero_votes, key=lambda tup: tup[1], reverse=True)


def percentage(num, denom):
    """Return the percentage string of a numerator to its denominator."""
    return '{:.4f}'.format((float(num) / denom) * 100)


def votes_with_percentages(vote_total, all_votes):
    """Return a list of parties along with their votes and percentage share."""
    return list(map(lambda kv: (kv[0], percentage(kv[1], vote_total)), all_votes))


def votes():
    """Compute zweitstimmen for all parties in the csv."""
    vs = get_csv_values()
    ps = parties()

    # contains `-` values for missing votes.
    unfiltered_values = list(map(lambda x: (x, second_votes(x, vs)), ps))

    cleansed_votes = cleanse_votes(unfiltered_values)
    vote_total = reduce(lambda acc, kv: acc + kv[1], cleansed_votes, 0)

    return votes_with_percentages(vote_total, cleansed_votes)


def total_below(vs, percent):
    """Return total votes below the provided percentage."""
    other_votes = list(filter(lambda x: float(x[1]) < percent, vs))
    return reduce(lambda acc, kv: acc + float(kv[1]), other_votes, 0)


def chart():
    """Display a bar chart of parties to the percentages of their votes."""
    vs = votes()

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
    vs = votes()

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


chart_with_labels()
