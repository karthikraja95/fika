import scipy as sc

def run_2sample_ttest(
    group1: str, group2: str, train_data, t_type: str, output_file, **kwargs
):

    data_group1 = train_data[group1].tolist()
    data_group2 = train_data[group2].tolist()

    if t_type == "ind":
        results = sc.stats.ttest_ind(
            data_group1, data_group2, nan_policy="omit", **kwargs
        )
    else:
        results = sc.stats.ttest_rel(data_group1, data_group2, nan_policy="omit",)

    return results