# UAE Health Insurance & Hospital Analytics: results

## 1. Claims fraud, waste and abuse (test: 13,106 claims, Sep-Dec 2025)

FWA share 6.6% of claims, AED 1,240,882 at risk. Audit capacity: 5% of claims (655).

| Review queue ordered by | Precision | FWA claims found | AED caught | Share of FWA AED | Net saving after review cost |
|---|---|---|---|---|---|
| Claim edits (rules) | 59.5% | 390 / 870 | 604,299 | 48.7% | 506,049 |
| **ML model + peer comparison, ranked by AED at risk** | **85.8%** | **562 / 870** | **1,086,574** | **87.6%** | **988,324** |
| ML model, ranked by probability only | 97.9% | 641 / 870 | 788,736 | 63.6% | 690,486 |

The rules flag 499 claims in total (77.0% precision). The model's queue is ranked by expected AED at risk (probability x amount), so reviewers spend their time where the money is.

Share of each scheme caught within capacity: duplicate 33%, excessive_tests 98%, phantom 36%, unbundling 100%, upcoding 99%

Example review-queue entries:

- C0011579 · O80 · 59400 · AED 9,982 → Unusual cost and pattern for this diagnosis
- D0000309 · O80 · 59400 · AED 9,173 → Near-identical claim submitted within 3 days
- C0006260 · O80 · 59400 · AED 8,978 → Unusual cost and pattern for this diagnosis
- C0021553 · S82.8 · 99284 73590 27786 82306 80053 85025 84443 86803 99285 · AED 8,685 → Lab tests far beyond what the diagnosis needs; Provider orders far more tests than peers
- C0004789 · S82.8 · 99284 73590 27786 99213 99213 99285 · AED 8,665 → Same service code billed more than once

## 2. Bed occupancy forecast, 7 days ahead (2025, 4 hospitals)

| Method | Mean error (beds) | Mean % error |
|---|---|---|
| **Admissions model + length-of-stay curve** | **17.82** | **7.1%** |
| Same as today | 20.07 | 8.4% |
| Average of the last 4 weeks | 20.65 | 8.6% |

High-occupancy alert (forecast ≥ 90% when actual ≥ 92%): recall 56%, precision 68% over 413 high-occupancy hospital-days.