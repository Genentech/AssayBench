from assaybench.dataset.dataset import AssayBenchDataset
from assaybench.benchmark.metrics import RankingMetrics

dataset_name = "biogrid"
novel_dataset_name = "LaTest"  # set to None if not using a novel dataset
split_type = "year"  # or "random"
fold = 0  # which fold to use in the given split type

ds = AssayBenchDataset(
                dataset_name=dataset_name,
                novel_dataset_name=novel_dataset_name,
                split_type=split_type,
                fold=fold,
            )


train,val,test, novel = ds.get_train_test_split()
print(f"Number of screens in train: {len(train)}")
print(f"Number of screens in val: {len(val)}")
print(f"Number of screens in test: {len(test)}")
print(f"Number of screens in novel dataset: {len(novel)}")

# Here we define out model: a function that take in a prompt and outputs a list of genes)
top_list = [ex["answer"].split(",") for ex in train]
top_list = [item.strip() for sublist in top_list for item in sublist]
# select top 100 most common answers
from collections import Counter
counter = Counter(top_list)
most_common = counter.most_common(100)
most_common_answers = [x[0] for x in most_common]
print(f"Most common answers: {most_common_answers}")

# our model
def top_100_answers(prompt):
    return most_common_answers

metric_fn = RankingMetrics(k_values=[10,100])

metrics = {ex["dataset_name"]: metric_fn.evaluate(top_100_answers(ex["question"]),
                                                  ex["relevance_genes"],
                                                  ex["relevance_scores"]) for ex in val}

adncg_at_100 = [m["adjusted_ndcg@100"] for m in metrics.values()]
print(f"Average adjusted nDCG@100: {sum(adncg_at_100)/len(adncg_at_100)}")

