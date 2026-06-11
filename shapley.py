"""
Monte Carlo Shapley Value Estimator
-------------------------------------
Computes approximate Shapley values for a single prediction
without using any explainability library.

What Shapley values actually are:
    From cooperative game theory. Each feature is a "player".
    Its Shapley value = its average marginal contribution
    across every possible ordering of features.

    Exact computation = 2^n subsets. For 50 features: 1 quadrillion operations.
    Monte Carlo fix: sample random orderings instead of exhausting all of them.
"""

import numpy as np


def mc_shapley(model, feature_names, patient, background, n_samples=512):
    """
    Estimates Shapley values for one patient via Monte Carlo sampling.

    Args:
        model       : trained sklearn model with predict_proba
        feature_names: list of feature names
        patient     : 1D numpy array of the patient's feature values
        background  : 2D numpy array of background dataset (used as reference distribution)
        n_samples   : number of random orderings to sample (higher = more accurate)

    Returns:
        dict of {feature_name: shapley_value}

    How it works:
        For each sampled ordering of features:
            1. Pick a random "coalition" — features that appear BEFORE feature i
            2. Build two synthetic patients:
               - with_i:    coalition features from patient, rest from a random background row
               - without_i: same but feature i also taken from background
            3. Marginal contribution = model(with_i) - model(without_i)
        Shapley value = mean marginal contribution across all sampled orderings.
    """
    n_features = len(feature_names)
    shapley_values = np.zeros(n_features)

    for _ in range(n_samples):
        # Random ordering of all features
        ordering = np.random.permutation(n_features)

        # Random background row as the "absent feature" substitute
        bg_row = background[np.random.randint(0, len(background))].copy()

        for pos, i in enumerate(ordering):
            # Coalition = all features that appear before i in this ordering
            coalition = ordering[:pos]

            # with_i: patient values for coalition + feature i, background for the rest
            with_i = bg_row.copy()
            with_i[coalition] = patient[coalition]
            with_i[i] = patient[i]

            # without_i: same coalition from patient, but feature i still from background
            without_i = bg_row.copy()
            without_i[coalition] = patient[coalition]

            # Marginal contribution of feature i in this ordering
            p_with    = model.predict_proba(with_i.reshape(1, -1))[0][1]
            p_without = model.predict_proba(without_i.reshape(1, -1))[0][1]

            shapley_values[i] += (p_with - p_without)

    # Average across all sampled orderings
    shapley_values /= n_samples

    return dict(zip(feature_names, np.round(shapley_values, 4)))


def display_shapley(patient_prob, shapley_dict, top_n=5):
    ranked = sorted(shapley_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]

    print("\n── Shapley Attribution (Monte Carlo, n=512) ──")
    print(f"  Base risk score: {patient_prob:.1%}")
    print(f"\n  Top {top_n} features by Shapley value:")
    for i, (feature, phi) in enumerate(ranked, 1):
        bar = "█" * int(abs(phi) * 200)
        direction = "↑" if phi > 0 else "↓"
        print(f"    {i}. {feature.replace('_',' ').title():<35} φ = {phi:+.4f}  {direction}  {bar}")
    print("\n  Sum of all Shapley values (should ≈ risk - base rate):")
    print(f"    {sum(shapley_dict.values()):+.4f}")
    print("──────────────────────────────────────────────\n")
