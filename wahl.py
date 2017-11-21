# coding=utf-8
# Written using python 3.6
import csv
import pandas as pd
import matplotlib.pyplot as plt
from pprint import pprint
from functools import reduce
from collections import Counter

plt.rcdefaults()


def fetch_rows(csv_path):
    """Open a CSV file and read its lines."""
    rows = []

    with open(csv_path, 'rt', encoding='utf8') as f:
        reader = csv.reader(f)
        for row in reader:
            rows.append(row)

    return rows


def get_csv_values(csv_path):
    """Return the semi-colon separated values in a csv as a list."""
    rows = fetch_rows(csv_path)

    # Remove column headings
    del rows[0]

    return list(map(lambda x: x[0].split(';'), rows))


def get_vote_values():
    return get_csv_values('./ergebnisse.csv')


def get_population_values():
    return get_csv_values('./population.csv')


def filter_not_parties(values):
    """Remove entries that are not for parties."""
    not_parties = ['Wahlberechtigte', 'Wähler', 'Ungültige', 'Gültige']
    return list(filter(lambda x: x[2] not in not_parties, values))


def values_at(csv_values, index):
    """Return all values at an index in the csv."""

    filtered = filter_not_parties(csv_values)

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
    """Aggregate values of a list second element in a 2-tuple."""
    return kv[0], reduce(lambda acc, x: acc + int(x), kv[1], 0)


def total(values):
    """Sum up values of all seconds in a list of 2-tuples."""
    return reduce(lambda acc, kv: acc + int(kv[1]), values, 0)


def lookup_alt_names(alternate_names, party):
    """Look up a party's alternate names."""
    maybe = list(filter(lambda x: x[0].upper() == party[0].upper(), alternate_names))

    if not maybe:
        return []

    return maybe[0][1]


def lookup_1st_value(values, match_text):
    """Return the first matching tuple of first values in a list of tuples."""
    maybe = list(filter(lambda x: x[0].upper() == match_text.upper(), values))

    if not maybe:
        return match_text, 0

    return maybe[0]


def merge_alt_names(alternate_names, alternates_with_votes, party):
    """Merge the votes of a party with votes of its known alternate names."""
    matched_alt_names = lookup_alt_names(alternate_names, party)
    matched_votes = list(map(lambda x: lookup_1st_value(alternates_with_votes, x), matched_alt_names))
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

    return list(map(lambda x: merge_alt_names(alternate_names, alts, x), originals))


def constituency_votes(constituency, values, vote_index):
    """Return the votes for the parties in a constituency."""
    const_vals = values_by(values, constituency, 1)
    votes = list(map(lambda x: (x[2], x[vote_index]), const_vals))
    merged_votes = merge_parties_alt_names(votes)

    return merged_votes


def constituencies_votes(state, values, vote_index):
    """Return the votes for the constituencies in a state."""
    state_vals = values_by(values, state, 0)
    constituencies = unique_values_at(state_vals, 1)

    return list(map(lambda x: (x, constituency_votes(x, state_vals, vote_index)), constituencies))


def cleanse_votes_by_constituencies(values, vote_index):
    """Return a list of states, where each state is a tuple of its id & a list of its constituents,
         & each constituent is a tuple of its id & a list of tuples of its parties & their votes.
         E.G.:
         [[('1',
               [('11',
                 [('CDU', 5000), ('DIE LINKE', 4000)]),
                ('15',
                 [('DIE LINKE', 6000), ('CDU', 3435)])])]]
         """

    real_parties = filter_not_parties(values)
    with_votes = list(filter(lambda x: x[vote_index] != '-', real_parties))
    states = unique_values_at(values, 0)
    states_votes = list(map(lambda x: (x, constituencies_votes(x, with_votes, vote_index)), states))

    return states_votes


def cleanse_first_votes_by_constituencies(values):
    return cleanse_votes_by_constituencies(values, 3)


def first_votes_by_constituencies():
    """Return parties in all states and constituencies with their 1st votes."""
    vs = get_vote_values()

    return cleanse_first_votes_by_constituencies(vs)


def higher(current, new):
    """Return the tuple with a higher value in its 2nd element among 2 tuples."""
    if int(new[1]) > int(current[1]):
        return new
    return current


def constituency_winner(parties):
    """Return the winner of a constituency."""
    return reduce(lambda acc, x: higher(acc, x), parties, ('', 0))


def state_constituency_winners(state_vals):
    """Return the winners of the constituencies in a state."""
    return list(map(lambda y: (y[0], constituency_winner(y[1])[0]), state_vals))


def direct_seat_winners():
    """Return the winners of the direktmandat for all constituencies in all states.
    E.g. [('1', [('11', 'CDU'), ('4', 'DIE LINKE')]), ('2', [('1', 'CDU'), ('10', 'SPD')])]"""
    votes = first_votes_by_constituencies()

    return list(map(lambda x: (x[0], state_constituency_winners(x[1])), votes))


def wins_per_party(constituency_winners):
    """Return the number of constituencies won by each party."""
    parties = list(map(lambda x: x[1], constituency_winners))
    return list(Counter(parties).items())


def states_direct_seats():
    """Return each party's share of wins in each state."""
    winners = direct_seat_winners()
    return list(map(lambda x: (x[0], wins_per_party(x[1])), winners))


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
    vs = get_vote_values()
    # party names are at the 3rd index.
    parties = unique_values_at(vs, 2)

    # contains `-` values for missing votes.
    unfiltered_values = list(map(lambda x: (x, party_second_votes(x, vs)), parties))

    cleansed_votes = cleanse_second_votes(unfiltered_values)
    vote_total = total(cleansed_votes)

    return cleansed_votes, vote_total


def cleanse_second_votes_by_constituencies(values):
    return cleanse_votes_by_constituencies(values, 4)


def second_votes_by_constituencies():
    """Return parties in all states and constituencies with their 2nd votes."""
    votes = get_vote_values()

    return cleanse_second_votes_by_constituencies(votes)


def add_if_party_matches(party, acc, party_votes):
    """Add vote if owner matches a party."""
    if party == party_votes[0]:
        return acc + int(party_votes[1])

    return acc


def second_vote_by_state(state_votes):
    """Return parties in a states with their 2nd votes."""
    votes = list(map(lambda x: x[1], state_votes))
    flat_votes = [item for sublist in votes for item in sublist]
    parties = set(list(map(lambda x: x[0], flat_votes)))

    return list(map(lambda x: (x, reduce(lambda acc, y: add_if_party_matches(x, acc, y), flat_votes, 0)), parties))


def second_votes_by_states():
    """Return parties in all states with their 2nd votes."""
    votes = second_votes_by_constituencies()

    return list(map(lambda x: (x[0], second_vote_by_state(x[1])), votes))


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


def compute_seats(total_seats, proportions):
    """Seat allocation algorithm."""
    total_proportions = total(proportions)
    divisor = round(float(total_proportions) / total_seats)

    while True:
        trial_seats = list(map(lambda x: (x[0], round(float(x[1]) / divisor)), proportions))
        total_trial_seats = total(trial_seats)

        if total_trial_seats != total_seats:
            if total_trial_seats > total_seats:
                divisor += 1
            else:
                divisor -= 1
            continue

        return trial_seats


def state_seat_distribution():
    """Allocate seats to each state based on population."""
    total_seats = 598
    state_pops = list(map(lambda x: (x[0], x[2]), get_population_values()))

    return compute_seats(total_seats, state_pops)


def compute_state_seats(state_votes, state_distribution, eligible_parties):
    """Compute no. of seats for parties in each state."""
    state, parties_votes = state_votes
    eligible_votes = list(filter(lambda x: x[0] in eligible_parties, parties_votes))
    _, total_seats = lookup_1st_value(state_distribution, state)

    return compute_seats(total_seats, eligible_votes)


def second_vote_seat_distribution():
    """Allocate seats based on zweitstimmen."""
    votes = second_votes()
    votes_by_state = second_votes_by_states()
    state_distribution = state_seat_distribution()

    percentages = votes_with_percentages(votes)
    above_five_percent = list(filter(lambda x: float(x[1]) >= 5, percentages))
    eligible_parties = list(map(lambda x: x[0], above_five_percent))

    return list(map(lambda x: (x[0], compute_state_seats(x, state_distribution, eligible_parties)), votes_by_state))


def display_seat_distribution():
    """Print the seat distribution in the Bundestag."""
    first = states_direct_seats()
    second = second_vote_seat_distribution()
    state_names = list(map(lambda x: (x[0], x[1]), get_population_values()))

    print('state;party;direct_seats;list_seats;ueberhang')
    for i in second:
        for j in first:
            # if it's the same state
            if i[0] == j[0]:
                for k in i[1]:
                    first_vote_counterpart = lookup_1st_value(j[1], k[0])
                    _, state_name = lookup_1st_value(state_names, i[0])

                    # Ueberhang is the difference when direct seats are more than list seats.
                    ueberhang = int(first_vote_counterpart[1]) - int(k[1])
                    if ueberhang < 0:
                        ueberhang = 0

                    print('{0};{1};{2};{3};{4}'.format(state_name, k[0], first_vote_counterpart[1], k[1], ueberhang))


display_seat_distribution()
