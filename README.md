# Naive Bayes Fraud Classifier — README

This document explains, in plain prose, how `naive_bayes_fraud.py` computes
its results. It's meant as a companion to the code — read this if you want
the explanation without reading Python.

The program is built entirely around one idea: **Bayes' theorem**.

```
posterior = (likelihood x prior) / evidence
```

- **posterior** — what we want to know: given everything we observed about a
  transaction, what's the probability it's fraud?
- **prior** — what we believed before looking at any details: how common is
  fraud overall?
- **likelihood** — if a transaction really were fraud, how typical would
  these exact details be?
- **evidence** — the overall probability of seeing these exact details at
  all, regardless of class. We never calculate this directly (it's the same
  number no matter which class we're scoring, so it can't change which class
  wins) — instead, we get the same effect by normalizing our scores at the
  end (dividing each class's score by the total of all classes' scores).

So in practice, every prediction in this program boils down to calculating
`prior x likelihood` for each class (fraud vs. not fraud), then seeing which
is bigger.

---

## Part 1 — How the toy example's results were computed

The toy example uses 10 made-up transactions (not real data) with two
features, chosen so the arithmetic is small enough to check by hand:

- **8 "not fraud" transactions**, with `amount` values 20, 22, 24, 26, 28, 30,
  32, 34, and `used_vpn` = True for 2 of them, False for the other 6.
- **2 "fraud" transactions**, with `amount` values 95 and 105, and
  `used_vpn` = False for both.

**Step 1 — Priors.** Out of 10 total transactions, 8 are "not fraud" and 2
are "fraud," so `prior(not fraud) = 8/10 = 0.8` and `prior(fraud) = 2/10 = 0.2`.

**Step 2 — Learn the numeric feature's shape.** For the `amount` feature, we
calculate the mean and variance separately for each class:
- not fraud: mean = 27.0, variance = 21.0
- fraud: mean = 100.0, variance = 25.0

**Step 3 — Likelihood of a new amount.** We classify a new, made-up
transaction: `amount = 93`, `used_vpn = True`. Plugging 93 into the Gaussian
(bell-curve) probability formula for each class's mean/variance gives a
likelihood of about `7.9 x 10^-47` for "not fraud" and about `0.030` for
"fraud" — unsurprising, since 93 is very close to the fraud group's mean of
100, and very far from the not-fraud group's mean of 27.

**Step 4 — Likelihood of the category feature, and the zero-count problem.**
For `used_vpn = True`, we just count: 2 of the 8 not-fraud examples used a
VPN (`2/8 = 0.25`), but **0 of the 2 fraud examples did** (`0/2 = 0.0`). That
zero is a real problem: it doesn't mean "impossible," it just means our tiny
sample of 2 fraud examples happened not to include a VPN user.

**Step 5 — Combine the likelihoods (the "naive" assumption).** Naive Bayes
assumes every feature is independent of every other feature, given the
class, so it multiplies their likelihoods together:
`combined likelihood = likelihood(amount) x likelihood(used_vpn)`. For the
fraud class, that's `0.030 x 0.0 = 0.0` — the zero from Step 4 wipes out the
entire fraud score, no matter how fraud-like the amount looked.

**Step 6 — Apply Bayes' theorem without smoothing.** `posterior = prior x
combined likelihood` gives a fraud score of exactly 0 and a not-fraud score
of about `1.6 x 10^-47`. Normalized, that's 0% fraud / 100% not fraud — the
model would classify this very fraud-looking transaction as **not fraud**,
entirely because of one sparse category value. That's the wrong call.

**Step 7 — Fix it with smoothing.** This technique is formally called
**Laplace smoothing** (or "add-one smoothing"). Instead of
`count / total`, we calculate `(count + smoothing_amount) / (total +
smoothing_amount x (possible_values + 1))`, which can never land on exactly
zero. With `smoothing_amount = 1`:
- not fraud: `(2+1)/(8+3) = 0.273`
- fraud: `(0+1)/(2+3) = 0.200` — no longer zero.

**Step 8 — Apply Bayes' theorem again, with smoothing.** Now the fraud
posterior score is about `1.2 x 10^-3`, compared to about `1.7 x 10^-47` for
not fraud. Normalized, that's effectively 100% fraud / 0% not fraud — the
correct classification. Smoothing didn't change the big picture (the amount
still overwhelmingly favors fraud); it just stopped one sparse category value
from incorrectly overriding everything else.

---

## Part 2 — How the real run's results were computed

The real classifier uses the exact same formula as Part 1, scaled up to the
full `credit_card_fraud_2026.csv` dataset (20,000 rows) and 24 real features
instead of 2 toy ones.

**Stratified train/test split.** The data is split into a training set
(80%, 16,000 rows) and a test set (20%, 4,000 rows). The split is
*stratified*: fraud and not-fraud rows are shuffled and divided separately,
so both the training set and the test set end up with about the same ~1.7%
fraud rate as the original data. Without this, a plain random split could
accidentally leave the test set with almost no fraud examples to check
against.

**Train/predict/evaluate boundary.** The real `is_fraud` answers for the test
rows already exist in the CSV (it's historical, already-labeled data) — we
aren't missing them. We deliberately keep them separate from the model
during prediction:
- *Training* gets both the training rows' features and their `is_fraud`
  answers, so it can calculate priors and likelihoods.
- *Prediction* gets only a row's features — never its label — so it's
  structurally impossible for the model to "cheat" by peeking at the answer.
- *Evaluation* is the only step that looks at the test rows' real answers,
  and only to compare them against what the model already guessed.

**Training.** For each of the 13 numeric columns, the program calculates a
mean and variance per class (same as Part 1's Step 2, just looped over more
columns). For each of the 11 categorical/boolean columns, it counts value
frequencies per class and applies the same smoothing formula from Part 1's
Step 7.

**Prediction.** For a new transaction, the program calculates
`prior x likelihood x likelihood x ...` for each class — but instead of
multiplying the plain numbers directly (as Part 1 does), it **adds the
logarithms** of the prior and each likelihood. This is mathematically the
same ranking (`log(a x b) = log(a) + log(b)`), but avoids "numerical
underflow": multiplying ~24 small decimals together would shrink the result
down to a number too small for the computer to represent accurately. The
final log-scores are converted back into normalized probabilities (e.g.
"62% estimated probability of fraud") for reporting.

**Evaluation metrics.** Because only about 1.7% of transactions are fraud, a
model that always guesses "not fraud" would still be about 98.3% "accurate"
while catching zero fraud — accuracy alone is misleading here. So results are
reported as:
- **Confusion matrix** — counts of true positives, true negatives, false
  positives (false alarms), and false negatives (missed fraud).
- **Precision** — of the transactions flagged as fraud, how many actually
  were fraud.
- **Recall** — of the transactions that really were fraud, how many were
  caught.
- **F1 score** — a single number balancing precision and recall.

On the test set, the model achieves roughly 97% accuracy, ~29% precision, and
~40% recall — meaning it catches about 2 out of every 5 real fraud cases,
with a bit under a third of its fraud alerts being correct. That's far better
than a model that never predicts fraud (0% recall), but far from perfect —
expected for a simple Naive Bayes model on an imbalanced dataset.

---

## Part 3 — Notes on the dataset's fields

| Column | Type in the model | Notes |
|---|---|---|
| `transaction_id` | *dropped* | Just a row label, not predictive of anything. |
| `amount_usd` | numeric | Transaction amount. |
| `merchant_category` | categorical | e.g. Restaurants, Electronics, Travel. |
| `card_type` | categorical | Visa, Mastercard, Amex, Discover, RuPay. |
| `auth_method` | categorical | How the transaction was authenticated. |
| `channel` | categorical | Online, POS, In-App, Contactless. |
| `device_type` | categorical | Phone, Mac, ATM Machine, etc. |
| `is_foreign_transaction` | categorical (boolean) | True/False treated as a 2-value category, not a number. |
| `hours_since_last_txn` | numeric | Time gap since the customer's previous transaction. |
| `txn_count_last_24h` | numeric | How many transactions this card made in the last day. |
| `distance_from_home_km` | numeric | Distance between transaction location and home. |
| `card_age_months` | numeric | How long the card has existed. |
| `customer_age` | numeric | Customer's age. |
| `account_balance_usd` | numeric | Balance at the time of the transaction. |
| `is_new_merchant` | categorical (boolean) | Whether the customer has used this merchant before. |
| `used_vpn` | categorical (boolean) | |
| `ip_country_mismatch` | categorical (boolean) | Whether the IP address's country doesn't match the card's. |
| `billing_shipping_mismatch` | categorical (boolean) | |
| `cvv_retry_count` | numeric | How many times the CVV was re-entered. |
| `velocity_score` | numeric | A pre-computed "how fast/unusual" score. |
| `time_of_day_hour` | numeric | Hour of day (0-23) the transaction happened. |
| `day_of_week` | numeric | Day of week the transaction happened. |
| `is_ai_generated_scam_attempt` | categorical (boolean) | Flag for AI-generated scam attempts. |
| `merchant_risk_score` | numeric | A pre-computed merchant risk score. |
| `prior_disputes` | numeric | Count of past disputes on the account. |
| `is_fraud` | *target* | What we're predicting: 0 = not fraud, 1 = fraud. |

**Things worth flagging:**

- **The dataset is heavily imbalanced**: only ~1.7% of transactions (339 out
  of 20,000) are fraud. This is exactly why precision/recall/F1 matter more
  than raw accuracy when judging the model (see Part 2 above).
- **Boolean-looking columns are treated as categories, not numbers.** A
  column like `used_vpn` holds "True"/"False," which are two distinct
  category labels — not quantities you'd want to average together the way
  you would with `amount_usd`.
- **A few columns look like unusually strong, almost "too convenient" fraud
  signals** for a synthetic/generated dataset — notably `merchant_risk_score`,
  `velocity_score`, and `is_ai_generated_scam_attempt`. Real-world fraud data
  is rarely labeled this cleanly; a model that performs well here may not
  generalize as well to messier, real-world data where such strong, almost
  pre-computed "fraud scores" usually aren't handed to you directly.
