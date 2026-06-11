import numpy as np


def mc_shapley(model, feature_names, patient, background, n_samples=512):
    """
    Monte Carlo Shapley value estimator.
    Instead of going through all 2^n feature subsets (impossible for 45 features),
    we sample random orderings and average the marginal contributions.
    Batches all predict_proba calls for speed.
    """
    n_features = len(feature_names)
    shapley_values = np.zeros(n_features)

    with_batch = []
    without_batch = []
    feature_indices = []

    for _ in range(n_samples):
        ordering = np.random.permutation(n_features)
        bg_row = background[np.random.randint(len(background))].copy()

        for pos, i in enumerate(ordering):
            coalition = ordering[:pos]

            with_i = bg_row.copy()
            with_i[coalition] = patient[coalition]
            with_i[i] = patient[i]

            without_i = bg_row.copy()
            without_i[coalition] = patient[coalition]

            with_batch.append(with_i)
            without_batch.append(without_i)
            feature_indices.append(i)

    # two big predict_proba calls instead of 2 * n_samples * n_features small ones
    all_rows = np.vstack(with_batch + without_batch)
    probs = model.predict_proba(all_rows)[:, 1]

    n_pairs = len(feature_indices)
    p_with    = probs[:n_pairs]
    p_without = probs[n_pairs:]

    for idx, feat_i in enumerate(feature_indices):
        shapley_values[feat_i] += p_with[idx] - p_without[idx]

    shapley_values /= n_samples
    return dict(zip(feature_names, np.round(shapley_values, 4)))


def display_shapley(patient_prob, shapley_dict, top_n=5):
    ranked = sorted(shapley_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]
    print(f"\nbase risk score: {patient_prob:.1%}")
    print(f"top {top_n} features driving this prediction:")
    print("-" * 45)
    for i, (feature, phi) in enumerate(ranked, 1):
        bar = "█" * int(abs(phi) * 200)
        direction = "↑ risk" if phi > 0 else "↓ risk"
        print(f"  {i}. {feature.replace('_', ' '):<30} {phi:+.4f}  {direction}  {bar}")
    print("-" * 45)
    print(f"  shapley sum: {sum(shapley_dict.values()):+.4f}  |  model output: {patient_prob:.4f}")
