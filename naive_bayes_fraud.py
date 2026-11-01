"""
Credit Card Fraud Detector — Naive Bayes Classifier (built from scratch)
=========================================================================

This program guesses whether a credit card transaction is fraud (is_fraud = 1)
or not fraud (is_fraud = 0). The whole program is really just one idea, used
over and over: BAYES' THEOREM.


THE CORE IDEA: BAYES' THEOREM
------------------------------
Bayes' theorem is a formula for updating a belief once new evidence shows up.
Written out, it looks like this:

               likelihood  x  prior
    posterior = ------------------------
                       evidence

In our fraud example, each word means:

    posterior   = what we actually want to know:
                  "given everything we observed about this transaction,
                  what's the probability it's fraud?"

    prior       = what we believed BEFORE looking at any details:
                  "out of all transactions, what fraction are normally fraud?"
                  (in our dataset, that's about 1.7%)

    likelihood  = "IF this transaction really were fraud, how typical would
                  these exact details (amount, location, device, etc.) be?"

    evidence    = the overall probability of seeing these exact details at
                  all, no matter which class the transaction belongs to.

We never calculate "evidence" directly. Here's why: we're going to calculate
a score for EVERY possible class (fraud and not fraud) and then see which
score is bigger. "Evidence" would be the exact same number no matter which
class we're scoring, so it can't change which class wins — it only matters
for turning our scores into real percentages. So instead, at the very end, we
"normalize": we divide each class's score by the total of all the scores.
That accomplishes the same thing as dividing by the evidence, and it turns our
raw scores into real probabilities that add up to 100%, like "73% chance this
is fraud."

So in practice, all we ever need to compute is:

    posterior score for a class   =   prior  x  likelihood

...for every class, then pick whichever class has the biggest score (and
normalize if we want an actual percentage instead of just a winner).


THE "NAIVE" PART: THE INDEPENDENCE ASSUMPTION
-----------------------------------------------
A real transaction has many details at once (amount, merchant, device, time
of day...). To get ONE likelihood number out of all those details, we
pretend every detail is independent of every other detail, once we already
know the class. That lets us just multiply the likelihood of each detail
together:

    likelihood of ALL details given class
        =  likelihood of detail 1 given class
         x likelihood of detail 2 given class
         x likelihood of detail 3 given class
         x  ...

This is the "naive" in "Naive Bayes" — it's not really true! In real life, a
transaction's dollar amount and its merchant category probably ARE related
to each other somewhat (a $4,000 purchase at "Groceries" is unusual in a way
a $4,000 purchase at "Electronics" is not). Pretending they're unrelated is a
simplification. The surprising thing is that Naive Bayes still tends to work
pretty well in practice even though this assumption isn't perfectly true.


TWO KINDS OF "LIKELIHOOD" MATH IN THIS PROGRAM
-------------------------------------------------
Our data has two different kinds of columns, so we calculate "likelihood" two
different ways:

1. NUMBER columns (like transaction amount) — we assume the numbers for each
   class roughly follow a bell-curve shape, and use the Gaussian (bell-curve)
   probability formula. This piece is called "Gaussian Naive Bayes."

2. CATEGORY columns (like card type, or True/False columns) — we just count
   how often each value shows up for each class, and turn those counts into
   probabilities. This piece is called "Categorical Naive Bayes," and it
   needs one extra trick (smoothing) that Part 1 below explains and
   demonstrates.


HOW THIS FILE IS ORGANIZED
-----------------------------
PART 1 below is a tiny, made-up example (not the real data) with numbers
small enough to check with a calculator. It exists purely to SEE Bayes'
theorem run, step by step, on numbers that can be verified by hand, before
trusting the big real version.

PART 2 runs the exact same formula — literally the same math — on the real
20,000-row dataset. The only real difference is a technical one (explained
in Part 2) about how the multiplication is done inside the computer.

PART 3 is a README.md file (written separately) that explains all of this
again in plain prose, for anyone who wants the explanation without reading
Python code.
"""

import math
import os
import random

import pandas as pd


# ===========================================================================
# Small math helper functions, used by both Part 1 and Part 2
# ===========================================================================

def calculate_mean(numbers):
    """Return the average of a list of numbers."""
    return sum(numbers) / len(numbers)


def calculate_variance(numbers, mean_value):
    """
    Return the variance of a list of numbers: a measure of how spread out the
    numbers are from their mean. A bigger variance means the numbers are more
    spread out; a smaller variance means they are bunched close together.
    """
    squared_differences = []
    for number in numbers:
        squared_differences.append((number - mean_value) ** 2)
    return sum(squared_differences) / len(numbers)


def gaussian_probability(x, mean_value, variance_value):
    """
    Compute how likely the value x is, ASSUMING it comes from a bell-curve
    (Gaussian / Normal distribution) with the given mean and variance.

    This is the standard Gaussian probability density formula:

        f(x) = (1 / sqrt(2 * pi * variance)) * e ^ ( -(x - mean)^2 / (2 * variance) )

    This formula doesn't need to be memorized to use it — just know that it
    returns a BIGGER number when x is close to the mean, and a number that
    shrinks very quickly the further x gets from the mean. This is exactly
    the "likelihood" piece of Bayes' theorem for a number-valued feature:
    "if this data point really belongs to this class, how typical is this
    particular value?"
    """
    exponent = -((x - mean_value) ** 2) / (2 * variance_value)
    coefficient = 1 / math.sqrt(2 * math.pi * variance_value)
    return coefficient * math.exp(exponent)


def apply_bayes_theorem(prior, likelihood):
    """
    This function IS Bayes' theorem (minus the "evidence" division, which we
    explained above we don't need in order to just compare classes).

        posterior score  =  prior  x  likelihood

    - "prior" is P(class): how common this class is, before looking at any
      details.
    - "likelihood" is P(details | class): how typical the observed details
      are, assuming this class is true.

    The number this function returns isn't a real probability by itself (it
    doesn't account for "evidence"), but comparing this score across classes
    tells us which class Bayes' theorem favors. We turn these scores into
    real probabilities later by normalizing (see normalize_scores_into_probabilities).
    """
    return prior * likelihood


def normalize_scores_into_probabilities(scores_by_class):
    """
    Take a dictionary of {class_label: raw_posterior_score} and turn it into
    real probabilities that add up to 1.0 (100%).

    This is standing in for dividing by "evidence" in Bayes' theorem: instead
    of calculating the evidence directly, we just divide each class's score
    by the total of ALL classes' scores. That produces the same end result -
    real, comparable probabilities - without ever needing to calculate
    "evidence" on its own.
    """
    total_of_all_scores = sum(scores_by_class.values())

    probabilities_by_class = {}
    for class_label in scores_by_class:
        if total_of_all_scores > 0:
            probabilities_by_class[class_label] = scores_by_class[class_label] / total_of_all_scores
        else:
            # Every class scored exactly 0 (can happen without smoothing, as
            # Part 1 demonstrates). There's no sensible percentage to report,
            # so we just split the probability evenly as a fallback.
            probabilities_by_class[class_label] = 1 / len(scores_by_class)

    return probabilities_by_class


# ===========================================================================
# PART 1: A tiny, made-up example that can be checked by hand with a calculator
# ===========================================================================

def run_tiny_bayes_theorem_example():
    """
    Walk through Bayes' theorem on a handful of made-up transactions, with
    small enough numbers that every step can be checked by hand. This uses
    NO real data from the CSV file - it's a teaching example only.

    The toy "dataset" has 10 made-up transactions and just 2 features:
      - amount:     a number (like a real transaction amount)
      - used_vpn:   True or False (like a real category feature)
    """
    print("=" * 70)
    print("PART 1: A tiny, hand-checkable Bayes' theorem example")
    print("=" * 70)

    # --- Our tiny made-up "training data" -----------------------------
    # 8 not-fraud transactions:
    not_fraud_amounts = [20, 22, 24, 26, 28, 30, 32, 34]
    not_fraud_used_vpn = [False, True, False, False, True, False, False, False]

    # 2 fraud transactions (on purpose, NEITHER of them used a VPN - this
    # will matter a lot in a moment):
    fraud_amounts = [95, 105]
    fraud_used_vpn = [False, False]

    number_of_not_fraud_rows = len(not_fraud_amounts)
    number_of_fraud_rows = len(fraud_amounts)
    total_rows = number_of_not_fraud_rows + number_of_fraud_rows

    print(f"\nToy training data: {number_of_not_fraud_rows} not-fraud rows, "
          f"{number_of_fraud_rows} fraud rows ({total_rows} total).")

    # --- Step 1: the PRIORS ---------------------------------------------
    # P(class) = how common each class is, before looking at any details.
    prior_not_fraud = number_of_not_fraud_rows / total_rows
    prior_fraud = number_of_fraud_rows / total_rows

    print("\nStep 1: Calculate the priors (P(class), before looking at any details)")
    print(f"  prior(not fraud) = {number_of_not_fraud_rows}/{total_rows} = {prior_not_fraud:.3f}")
    print(f"  prior(fraud)     = {number_of_fraud_rows}/{total_rows} = {prior_fraud:.3f}")

    # --- Step 2: learn the numeric feature's shape for each class -------
    # For the "amount" feature we need a mean and variance for each class,
    # so we can measure "how typical" a new amount is for that class.
    mean_amount_not_fraud = calculate_mean(not_fraud_amounts)
    variance_amount_not_fraud = calculate_variance(not_fraud_amounts, mean_amount_not_fraud)
    mean_amount_fraud = calculate_mean(fraud_amounts)
    variance_amount_fraud = calculate_variance(fraud_amounts, mean_amount_fraud)

    print("\nStep 2: Learn the shape of the 'amount' feature for each class")
    print(f"  not fraud: mean = {mean_amount_not_fraud:.1f}, variance = {variance_amount_not_fraud:.1f}")
    print(f"  fraud:     mean = {mean_amount_fraud:.1f}, variance = {variance_amount_fraud:.1f}")

    # --- A new transaction we want to classify --------------------------
    new_transaction_amount = 93
    new_transaction_used_vpn = True

    print(f"\nNow let's classify a brand-new transaction: "
          f"amount = {new_transaction_amount}, used_vpn = {new_transaction_used_vpn}")

    # --- Step 3: the LIKELIHOOD of the amount, for each class -----------
    likelihood_amount_given_not_fraud = gaussian_probability(
        new_transaction_amount, mean_amount_not_fraud, variance_amount_not_fraud
    )
    likelihood_amount_given_fraud = gaussian_probability(
        new_transaction_amount, mean_amount_fraud, variance_amount_fraud
    )

    print("\nStep 3: Calculate the likelihood of amount = 93 under each class")
    print(f"  likelihood(amount=93 | not fraud) = {likelihood_amount_given_not_fraud:.3e}")
    print(f"  likelihood(amount=93 | fraud)     = {likelihood_amount_given_fraud:.3e}")
    print("  (93 is very close to the fraud group's mean of 100, and very far")
    print("   from the not-fraud group's mean of 27 - so this detail alone")
    print("   points strongly toward fraud.)")

    # --- Step 4: the LIKELIHOOD of used_vpn, for each class -------------
    # This is a category feature, so "likelihood" just means counting.
    count_vpn_true_not_fraud = sum(1 for value in not_fraud_used_vpn if value is True)
    count_vpn_true_fraud = sum(1 for value in fraud_used_vpn if value is True)

    likelihood_vpn_given_not_fraud_NO_SMOOTHING = count_vpn_true_not_fraud / number_of_not_fraud_rows
    likelihood_vpn_given_fraud_NO_SMOOTHING = count_vpn_true_fraud / number_of_fraud_rows

    print("\nStep 4: Calculate the likelihood of used_vpn = True under each class")
    print("  (by simply counting how often used_vpn = True showed up in training)")
    print(f"  likelihood(used_vpn=True | not fraud) = {count_vpn_true_not_fraud}/{number_of_not_fraud_rows} "
          f"= {likelihood_vpn_given_not_fraud_NO_SMOOTHING:.3f}")
    print(f"  likelihood(used_vpn=True | fraud)     = {count_vpn_true_fraud}/{number_of_fraud_rows} "
          f"= {likelihood_vpn_given_fraud_NO_SMOOTHING:.3f}   <-- exactly ZERO!")
    print("  Neither of our 2 fraud examples happened to use a VPN, so as far")
    print("  as our training data is concerned, 'fraud AND used a VPN' never happened.")

    # --- Step 5: combine the two likelihoods (THE NAIVE ASSUMPTION) -----
    # This multiplication is exactly where the "naive" independence
    # assumption is being used: we're treating amount and used_vpn as if
    # they have nothing to do with each other, once we know the class, so we
    # can just multiply their individual likelihoods together.
    combined_likelihood_not_fraud_NO_SMOOTHING = (
        likelihood_amount_given_not_fraud * likelihood_vpn_given_not_fraud_NO_SMOOTHING
    )
    combined_likelihood_fraud_NO_SMOOTHING = (
        likelihood_amount_given_fraud * likelihood_vpn_given_fraud_NO_SMOOTHING
    )

    print("\nStep 5: Combine both likelihoods by multiplying them together")
    print("  (this is the 'naive' independence assumption in action - we're")
    print("   treating amount and used_vpn as unrelated, given the class)")
    print(f"  combined likelihood (not fraud) = {combined_likelihood_not_fraud_NO_SMOOTHING:.3e}")
    print(f"  combined likelihood (fraud)     = {combined_likelihood_fraud_NO_SMOOTHING:.3e}   <-- ZERO!")
    print("  Even though the amount (93) looked VERY fraud-like, multiplying by")
    print("  a likelihood of exactly 0 wipes out the entire fraud score.")

    # --- Step 6: apply Bayes' theorem (without smoothing) ----------------
    posterior_not_fraud_NO_SMOOTHING = apply_bayes_theorem(
        prior_not_fraud, combined_likelihood_not_fraud_NO_SMOOTHING
    )
    posterior_fraud_NO_SMOOTHING = apply_bayes_theorem(
        prior_fraud, combined_likelihood_fraud_NO_SMOOTHING
    )

    probabilities_NO_SMOOTHING = normalize_scores_into_probabilities({
        "not fraud": posterior_not_fraud_NO_SMOOTHING,
        "fraud": posterior_fraud_NO_SMOOTHING,
    })

    print("\nStep 6: Apply Bayes' theorem (posterior = prior x likelihood), WITHOUT smoothing")
    print(f"  posterior score (not fraud) = {prior_not_fraud:.3f} x {combined_likelihood_not_fraud_NO_SMOOTHING:.3e} "
          f"= {posterior_not_fraud_NO_SMOOTHING:.3e}")
    print(f"  posterior score (fraud)     = {prior_fraud:.3f} x {combined_likelihood_fraud_NO_SMOOTHING:.3e} "
          f"= {posterior_fraud_NO_SMOOTHING:.3e}")
    print(f"  Normalized probabilities -> not fraud: {probabilities_NO_SMOOTHING['not fraud']:.1%}, "
          f"fraud: {probabilities_NO_SMOOTHING['fraud']:.1%}")
    if posterior_fraud_NO_SMOOTHING > posterior_not_fraud_NO_SMOOTHING:
        print("  WITHOUT smoothing, the model would classify this as: FRAUD")
    else:
        print("  WITHOUT smoothing, the model would classify this as: NOT FRAUD")
        print("  ...which is the WRONG call, caused entirely by one category")
        print("  value (used_vpn=True) that simply never showed up in our tiny")
        print("  fraud training set.")

    # --- Step 7: fix the zero-count problem with smoothing ---------------
    # This small fix is sometimes called "Laplace smoothing" or "add-one
    # smoothing" in a textbook - but all it really means is: pretend we
    # saw a few extra examples of every possible value,
    # so a probability is never calculated as exactly zero.
    smoothing_amount = 1
    number_of_possible_vpn_values = 2  # True and False

    likelihood_vpn_given_not_fraud_SMOOTHED = (
        (count_vpn_true_not_fraud + smoothing_amount)
        / (number_of_not_fraud_rows + smoothing_amount * (number_of_possible_vpn_values + 1))
    )
    likelihood_vpn_given_fraud_SMOOTHED = (
        (count_vpn_true_fraud + smoothing_amount)
        / (number_of_fraud_rows + smoothing_amount * (number_of_possible_vpn_values + 1))
    )

    print("\nStep 7: Fix the zero with smoothing (formally called 'Laplace smoothing')")
    print("  Instead of count / total, we calculate (count + smoothing_amount) /")
    print("  (total + smoothing_amount x (possible_values + 1)), which can never land on exactly zero.")
    print(f"  smoothed likelihood(used_vpn=True | not fraud) = "
          f"({count_vpn_true_not_fraud}+{smoothing_amount})/({number_of_not_fraud_rows}+{smoothing_amount}x3) "
          f"= {likelihood_vpn_given_not_fraud_SMOOTHED:.3f}")
    print(f"  smoothed likelihood(used_vpn=True | fraud)     = "
          f"({count_vpn_true_fraud}+{smoothing_amount})/({number_of_fraud_rows}+{smoothing_amount}x3) "
          f"= {likelihood_vpn_given_fraud_SMOOTHED:.3f}   <-- no longer zero!")

    combined_likelihood_not_fraud_SMOOTHED = (
        likelihood_amount_given_not_fraud * likelihood_vpn_given_not_fraud_SMOOTHED
    )
    combined_likelihood_fraud_SMOOTHED = (
        likelihood_amount_given_fraud * likelihood_vpn_given_fraud_SMOOTHED
    )

    posterior_not_fraud_SMOOTHED = apply_bayes_theorem(
        prior_not_fraud, combined_likelihood_not_fraud_SMOOTHED
    )
    posterior_fraud_SMOOTHED = apply_bayes_theorem(
        prior_fraud, combined_likelihood_fraud_SMOOTHED
    )

    probabilities_SMOOTHED = normalize_scores_into_probabilities({
        "not fraud": posterior_not_fraud_SMOOTHED,
        "fraud": posterior_fraud_SMOOTHED,
    })

    print("\nStep 8: Apply Bayes' theorem again, now WITH smoothing")
    print(f"  posterior score (not fraud) = {posterior_not_fraud_SMOOTHED:.3e}")
    print(f"  posterior score (fraud)     = {posterior_fraud_SMOOTHED:.3e}")
    print(f"  Normalized probabilities -> not fraud: {probabilities_SMOOTHED['not fraud']:.1%}, "
          f"fraud: {probabilities_SMOOTHED['fraud']:.1%}")
    if posterior_fraud_SMOOTHED > posterior_not_fraud_SMOOTHED:
        print("  WITH smoothing, the model classifies this as: FRAUD")
        print("  This is the correct call - smoothing stopped one sparse category")
        print("  value from overriding what the amount feature was clearly telling us.")
    else:
        print("  WITH smoothing, the model classifies this as: NOT FRAUD")

    print()


# ===========================================================================
# PART 2: The real classifier, trained on the full 20,000-row dataset
# ===========================================================================
#
# Everything below runs the EXACT SAME Bayes' theorem formula as Part 1
# above (posterior = prior x likelihood, likelihoods multiplied together
# feature by feature). The only real difference is explained in the
# NaiveBayesFraudClassifier.predict_one_with_probability method below: with
# about 24 features instead of 2, multiplying that many small decimals
# together would shrink the result down to a number too small for the
# computer to represent (this is called "numerical underflow"), so we add
# LOGARITHMS of the likelihoods instead of multiplying the plain likelihoods.
# log(a x b) = log(a) + log(b), so this gives the identical ranking of which
# class wins, without the underflow problem.
# ===========================================================================

# Build the CSV's path from this script's own location (not the current
# working directory), so the program finds the file correctly no matter
# which folder it gets run from (e.g. running it from VS Code's "Run"
# button vs. from a terminal opened somewhere else).
SCRIPT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
CSV_FILE_NAME = os.path.join(SCRIPT_DIRECTORY, "credit_card_fraud_2026.csv")

ID_COLUMN = "transaction_id"
TARGET_COLUMN = "is_fraud"

NUMERIC_COLUMNS = [
    "amount_usd",
    "hours_since_last_txn",
    "txn_count_last_24h",
    "distance_from_home_km",
    "card_age_months",
    "customer_age",
    "account_balance_usd",
    "cvv_retry_count",
    "velocity_score",
    "time_of_day_hour",
    "day_of_week",
    "merchant_risk_score",
    "prior_disputes",
]

CATEGORY_COLUMNS = [
    "merchant_category",
    "card_type",
    "auth_method",
    "channel",
    "device_type",
    "is_foreign_transaction",
    "is_new_merchant",
    "used_vpn",
    "ip_country_mismatch",
    "billing_shipping_mismatch",
    "is_ai_generated_scam_attempt",
]


def load_dataset(csv_file_name):
    """
    Read the CSV file and split it into three plain Python lists:

      numeric_rows   -> a list where each item is a list of numbers
      category_rows  -> a list where each item is a list of strings
      labels         -> a list of 0s and 1s (is_fraud for each row)

    We use pandas only to make reading the CSV file easy. Once it's loaded,
    we convert everything into plain Python lists so the rest of the program
    doesn't require knowing pandas to follow along.
    """
    table = pd.read_csv(csv_file_name)
    table = table.drop(columns=[ID_COLUMN])

    labels = table[TARGET_COLUMN].tolist()
    numeric_rows = table[NUMERIC_COLUMNS].values.tolist()

    # Convert category columns to strings so True/False values and text
    # values are both treated the same simple way: as category labels.
    category_rows = table[CATEGORY_COLUMNS].astype(str).values.tolist()

    return numeric_rows, category_rows, labels


def split_into_train_and_test(numeric_rows, category_rows, labels,
                               test_fraction=0.2, random_seed=42):
    """
    Split the dataset into a training set (used to teach the model) and a
    test set (used to check how well the model learned), while keeping the
    fraud / not-fraud ratio about the same in both sets. This is called a
    "stratified" split.

    Why stratify? Only about 1.7% of our transactions are fraud. If we split
    totally randomly, we might get unlucky and end up with very few (or even
    zero) fraud examples in the test set, which would make it impossible to
    fairly judge how good the model is at catching fraud.

    IMPORTANT: splitting the data is the ONLY place the real is_fraud answers
    get used before evaluation. Once we hand the test rows to the model's
    predict() method below, we only ever pass it the FEATURES
    (numeric_row, category_row) - never the label. The real answers for the
    test rows are set aside here and not looked at again until the Evaluate
    step at the very end, when we check the model's guesses against them.
    """
    random.seed(random_seed)

    fraud_indices = []
    not_fraud_indices = []
    for index in range(len(labels)):
        if labels[index] == 1:
            fraud_indices.append(index)
        else:
            not_fraud_indices.append(index)

    random.shuffle(fraud_indices)
    random.shuffle(not_fraud_indices)

    number_of_fraud_test_rows = int(round(len(fraud_indices) * test_fraction))
    number_of_not_fraud_test_rows = int(round(len(not_fraud_indices) * test_fraction))

    test_indices = (
        fraud_indices[:number_of_fraud_test_rows]
        + not_fraud_indices[:number_of_not_fraud_test_rows]
    )
    train_indices = (
        fraud_indices[number_of_fraud_test_rows:]
        + not_fraud_indices[number_of_not_fraud_test_rows:]
    )

    random.shuffle(train_indices)
    random.shuffle(test_indices)

    def build_subset(index_list):
        numeric_subset = [numeric_rows[i] for i in index_list]
        category_subset = [category_rows[i] for i in index_list]
        label_subset = [labels[i] for i in index_list]
        return numeric_subset, category_subset, label_subset

    numeric_train, category_train, labels_train = build_subset(train_indices)
    numeric_test, category_test, labels_test = build_subset(test_indices)

    return (numeric_train, category_train, labels_train,
            numeric_test, category_test, labels_test)


class NaiveBayesFraudClassifier:
    """
    A Naive Bayes classifier that runs the exact Bayes' theorem steps shown
    in Part 1 above, scaled up to 24 real features across 20,000 rows:

      1. Learn a prior for each class (fit).
      2. Learn a likelihood "shape" for every feature, for each class (fit).
      3. For a new transaction, combine prior x likelihood x likelihood x ...
         for each class (predict), and pick the bigger score.
    """

    def __init__(self, smoothing_amount=1, variance_smoothing=1e-9):
        # smoothing_amount is the same idea demonstrated in Part 1 (there
        # called "formally Laplace / add-one smoothing"): it stops a
        # category value that never appeared in training from forcing a
        # likelihood of exactly zero.
        self.smoothing_amount = smoothing_amount

        # variance_smoothing is a tiny number added to every variance so we
        # never accidentally divide by zero in the Gaussian formula.
        self.variance_smoothing = variance_smoothing

        self.class_labels = []
        self.prior_probability = {}             # class -> plain P(class), e.g. 0.017
        self.numeric_feature_stats = {}          # class -> list of (mean, variance)
        self.category_feature_likelihoods = []   # list (per feature) of {class: {value: probability}}
        self.category_feature_unseen_likelihood = []  # list (per feature) of {class: probability_for_unknown_value}

    def fit(self, numeric_rows, category_rows, labels):
        """
        Train the classifier. "Training" a Naive Bayes model just means
        calculating the priors and likelihoods from the training data -
        there's no complicated search/optimization loop like in other
        machine learning algorithms.
        """
        self.class_labels = sorted(set(labels))

        self._learn_priors(labels)
        self._learn_numeric_likelihoods(numeric_rows, labels)
        self._learn_category_likelihoods(category_rows, labels)

    def _learn_priors(self, labels):
        """Calculate P(class) for each class - exactly Part 1's Step 1."""
        total_rows = len(labels)
        for class_label in self.class_labels:
            count_for_this_class = 0
            for label in labels:
                if label == class_label:
                    count_for_this_class += 1
            self.prior_probability[class_label] = count_for_this_class / total_rows

    def _learn_numeric_likelihoods(self, numeric_rows, labels):
        """
        For every numeric feature, and for every class, calculate the mean
        and variance of that feature - exactly Part 1's Step 2, just looped
        over 13 numeric columns instead of 1.
        """
        number_of_numeric_features = len(numeric_rows[0])

        for class_label in self.class_labels:
            rows_for_this_class = []
            for row_index in range(len(numeric_rows)):
                if labels[row_index] == class_label:
                    rows_for_this_class.append(numeric_rows[row_index])

            stats_for_each_feature = []
            for feature_index in range(number_of_numeric_features):
                values_in_this_column = []
                for row in rows_for_this_class:
                    values_in_this_column.append(row[feature_index])

                mean_value = calculate_mean(values_in_this_column)
                variance_value = calculate_variance(values_in_this_column, mean_value)

                # Add a tiny amount so a feature that happens to have the
                # exact same value for every row in a class never produces
                # a variance of exactly 0 (which would crash the Gaussian
                # formula via division by zero).
                variance_value = variance_value + self.variance_smoothing

                stats_for_each_feature.append((mean_value, variance_value))

            self.numeric_feature_stats[class_label] = stats_for_each_feature

    def _learn_category_likelihoods(self, category_rows, labels):
        """
        For every categorical feature, and for every class, count how often
        each possible value shows up, then turn those counts into smoothed
        probabilities - exactly Part 1's Steps 4 and 7, looped over all 11
        categorical/boolean columns instead of 1.
        """
        number_of_category_features = len(category_rows[0])

        for feature_index in range(number_of_category_features):
            all_values_in_this_column = set()
            for row in category_rows:
                all_values_in_this_column.add(row[feature_index])
            number_of_possible_values = len(all_values_in_this_column)

            likelihoods_for_this_feature = {}
            unseen_likelihood_for_this_feature = {}

            for class_label in self.class_labels:
                value_counts = {}
                for value in all_values_in_this_column:
                    value_counts[value] = 0

                total_rows_for_this_class = 0
                for row_index in range(len(category_rows)):
                    if labels[row_index] == class_label:
                        value = category_rows[row_index][feature_index]
                        value_counts[value] += 1
                        total_rows_for_this_class += 1

                # Same smoothing formula demonstrated by hand in Part 1:
                #   (count + smoothing_amount) /
                #   (total + smoothing_amount * (possible_values + 1))
                denominator = (
                    total_rows_for_this_class
                    + self.smoothing_amount * (number_of_possible_values + 1)
                )

                likelihoods_for_this_value = {}
                for value in all_values_in_this_column:
                    likelihoods_for_this_value[value] = (
                        (value_counts[value] + self.smoothing_amount) / denominator
                    )

                likelihoods_for_this_feature[class_label] = likelihoods_for_this_value
                unseen_likelihood_for_this_feature[class_label] = self.smoothing_amount / denominator

            self.category_feature_likelihoods.append(likelihoods_for_this_feature)
            self.category_feature_unseen_likelihood.append(unseen_likelihood_for_this_feature)

    def predict_one_with_probability(self, numeric_row, category_row):
        """
        Classify a single transaction, returning BOTH the predicted class
        and a dictionary of normalized probabilities (e.g. {"0": 0.93,
        "1": 0.07}) so predictions can be reported as "93% likely not
        fraud" instead of just a bare label.

        As explained at the top of the file, we add LOGARITHMS of the prior
        and each likelihood instead of multiplying the plain numbers
        together. This is mathematically the same ranking as Part 1's plain
        multiplication (log(a x b) = log(a) + log(b)), just safe from
        numerical underflow with this many features.
        """
        log_score_by_class = {}

        for class_label in self.class_labels:
            # Start with log(prior) - same prior from Part 1's Step 1.
            log_score = math.log(self.prior_probability[class_label])

            # Add log(likelihood) for every numeric feature - same Gaussian
            # formula from Part 1's Step 3.
            numeric_stats_for_this_class = self.numeric_feature_stats[class_label]
            for feature_index in range(len(numeric_row)):
                value = numeric_row[feature_index]
                mean_value, variance_value = numeric_stats_for_this_class[feature_index]
                likelihood = gaussian_probability(value, mean_value, variance_value)

                # Guard against log(0) for an extremely unlikely value.
                if likelihood <= 0:
                    likelihood = 1e-300
                log_score += math.log(likelihood)

            # Add log(likelihood) for every categorical feature - same
            # counting + smoothing from Part 1's Steps 4 and 7.
            for feature_index in range(len(category_row)):
                value = category_row[feature_index]
                known_likelihoods = self.category_feature_likelihoods[feature_index][class_label]
                if value in known_likelihoods:
                    likelihood = known_likelihoods[value]
                else:
                    # A category value that never showed up anywhere during
                    # training - fall back to the smoothed "unseen" value.
                    likelihood = self.category_feature_unseen_likelihood[feature_index][class_label]
                log_score += math.log(likelihood)

            log_score_by_class[class_label] = log_score

        # Convert the log-scores back into real, normalized probabilities.
        # We subtract the largest log-score from every log-score first - a
        # standard trick that keeps math.exp() from overflowing, without
        # changing the final normalized probabilities at all.
        largest_log_score = max(log_score_by_class.values())
        unnormalized_scores = {}
        for class_label in log_score_by_class:
            unnormalized_scores[class_label] = math.exp(log_score_by_class[class_label] - largest_log_score)

        probabilities = normalize_scores_into_probabilities(unnormalized_scores)

        predicted_class = max(probabilities, key=probabilities.get)
        return predicted_class, probabilities

    def predict(self, numeric_rows, category_rows):
        """Predict the class for every row in a list of transactions."""
        predictions = []
        for row_index in range(len(numeric_rows)):
            predicted_class, _probabilities = self.predict_one_with_probability(
                numeric_rows[row_index], category_rows[row_index]
            )
            predictions.append(predicted_class)
        return predictions


# ===========================================================================
# Evaluation: measuring how good the model is
# ===========================================================================

def calculate_confusion_matrix(true_labels, predicted_labels):
    """
    Compare the true labels to the model's predictions and count up four
    numbers:

      true_positives  -> model said fraud, and it really was fraud
      true_negatives  -> model said not fraud, and it really wasn't
      false_positives -> model said fraud, but it actually wasn't ("false alarm")
      false_negatives -> model said not fraud, but it actually was ("missed fraud")
    """
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0

    for index in range(len(true_labels)):
        actual = true_labels[index]
        predicted = predicted_labels[index]

        if actual == 1 and predicted == 1:
            true_positives += 1
        elif actual == 0 and predicted == 0:
            true_negatives += 1
        elif actual == 0 and predicted == 1:
            false_positives += 1
        elif actual == 1 and predicted == 0:
            false_negatives += 1

    return true_positives, true_negatives, false_positives, false_negatives


def print_evaluation_report(title, true_labels, predicted_labels):
    """
    Print accuracy, precision, recall, and F1-score for a set of
    predictions, along with the confusion matrix they came from.

    Why not just use accuracy? Only about 1.7% of our transactions are
    fraud. A lazy model that always predicts "not fraud" would be right
    98.3% of the time and look great by accuracy alone, while being
    completely useless at catching fraud. So we also look at:

      precision -> of the transactions we flagged as fraud, how many
                   actually were fraud?
      recall    -> of the transactions that really were fraud, how many
                   did we catch?
      F1-score  -> a single number balancing precision and recall
    """
    true_positives, true_negatives, false_positives, false_negatives = \
        calculate_confusion_matrix(true_labels, predicted_labels)

    total = len(true_labels)
    accuracy = (true_positives + true_negatives) / total

    if (true_positives + false_positives) > 0:
        precision = true_positives / (true_positives + false_positives)
    else:
        precision = 0.0

    if (true_positives + false_negatives) > 0:
        recall = true_positives / (true_positives + false_negatives)
    else:
        recall = 0.0

    if (precision + recall) > 0:
        f1_score = 2 * precision * recall / (precision + recall)
    else:
        f1_score = 0.0

    print()
    print(f"--- {title} ---")
    print("Confusion matrix (rows = actual, columns = predicted):")
    print("                 predicted: 0   predicted: 1")
    print(f"  actual: 0       {true_negatives:>10}   {false_positives:>13}")
    print(f"  actual: 1       {false_negatives:>10}   {true_positives:>13}")
    print(f"Accuracy:  {accuracy:.4f}  (misleading on its own - only ~1.7% of rows are fraud)")
    print(f"Precision: {precision:.4f}  (of the fraud alerts we raised, how many were real)")
    print(f"Recall:    {recall:.4f}  (of the real fraud cases, how many we actually caught)")
    print(f"F1 score:  {f1_score:.4f}  (balance between precision and recall)")


def print_example_predictions(model, numeric_rows, category_rows, labels, how_many=5):
    """
    Print a handful of individual predictions, showing the normalized
    probability Bayes' theorem produced and the real answer side by side -
    useful for making the "posterior probability" idea feel concrete instead
    of abstract.
    """
    print(f"\n--- {how_many} example test-set predictions ---")
    for row_index in range(how_many):
        predicted_class, probabilities = model.predict_one_with_probability(
            numeric_rows[row_index], category_rows[row_index]
        )
        fraud_probability = probabilities[1]
        actual_label = labels[row_index]
        actual_text = "FRAUD" if actual_label == 1 else "not fraud"
        predicted_text = "FRAUD" if predicted_class == 1 else "not fraud"
        print(f"  Transaction {row_index + 1}: estimated {fraud_probability:.1%} probability of fraud "
              f"-> predicted {predicted_text} (actual: {actual_text})")


# ===========================================================================
# Put it all together
# ===========================================================================

def main():
    # Part 1 first: a small, hand-checkable demonstration of Bayes' theorem.
    run_tiny_bayes_theorem_example()

    # Part 2: the same formula, run for real on the full dataset.
    print("=" * 70)
    print("PART 2: The real classifier, trained on the full dataset")
    print("=" * 70)

    print(f"\nLoading data from {CSV_FILE_NAME} ...")
    numeric_rows, category_rows, labels = load_dataset(CSV_FILE_NAME)

    print("Splitting into training and test sets...")
    (numeric_train, category_train, labels_train,
     numeric_test, category_test, labels_test) = split_into_train_and_test(
        numeric_rows, category_rows, labels, test_fraction=0.2, random_seed=42
    )

    fraud_rate_train = sum(labels_train) / len(labels_train)
    fraud_rate_test = sum(labels_test) / len(labels_test)
    print(f"Training rows: {len(labels_train)}  (fraud rate: {fraud_rate_train:.2%})")
    print(f"Test rows:     {len(labels_test)}  (fraud rate: {fraud_rate_test:.2%})")

    print("\nTraining the Naive Bayes classifier (learning priors and likelihoods)...")
    model = NaiveBayesFraudClassifier(smoothing_amount=1, variance_smoothing=1e-9)
    model.fit(numeric_train, category_train, labels_train)

    print("Making predictions on the training set and the test set...")
    train_predictions = model.predict(numeric_train, category_train)
    test_predictions = model.predict(numeric_test, category_test)

    print_evaluation_report("Training set results", labels_train, train_predictions)
    print_evaluation_report("Test set results", labels_test, test_predictions)

    print_example_predictions(model, numeric_test, category_test, labels_test, how_many=5)


if __name__ == "__main__":
    main()
