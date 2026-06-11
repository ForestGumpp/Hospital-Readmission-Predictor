import numpy as np

# rough monte carlo shapley implementation
# instead of going through all 2^n feature subsets (impossible for 45 features)
# we just sample random orderings and average the marginal contributions

def mc_shapley(model, feature_names, patient, background, n_samples=512):
    n_features = len(feature_names)
    shapley_values = np.zeros(n_features)

    for _ in range(n_samples):
        ordering = np.random.permutation(n_features)
        bg_row = background[np.random.randint(0, len(background))].copy()

        for pos, i in enumerate(ordering):
            coalition = ordering[:pos]

            # with feature i included
            with_i = bg_row.copy()
            with_i[coalition] = patient[coalition]
            with_i[i] = patient[i]

            # without feature i (still from background)
            without_i = bg_row.copy()
            without_i[coalition] = patient[coalition]

            p_with = model.predict_proba(with_i.reshape(1, -1))[0][1]
            p_without = model.predict_proba(without_i.reshape(1, -1))[0][1]

            shapley_values[i] += (p_with - p_without)

    shapley_values /= n_samples
    return dict(zip(feature_names, np.round(shapley_values, 4)))


def display_shapley(patient_prob, shapley_dict, top_n=5):
    ranked = sorted(shapley_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]

    print(f"\nbase risk score: {patient_prob:.1%}")
    print(f"top {top_n} features:")
    for i, (feature, phi) in enumerate(ranked, 1):
        bar = "#" * int(abs(phi) * 200)
        direction = "up" if phi > 0 else "down"
        print(f"  {i}. {feature.replace('_',' ')}: {phi:+.4f} ({direction}) {bar}")

    print(f"\nsum of shapley values: {sum(shapley_dict.values()):+.4f}")
