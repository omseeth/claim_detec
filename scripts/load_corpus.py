"""
This script helps with loading the corpus. Files should be stored in /data/[
file].
"""
from brat_parser import get_entities_relations_attributes_groups
import pandas as pd
from pathlib import Path


class StabGurevychCorpus:
    """Load corpus files to pandas frame. Class object can be used for
    further processing."""

    def __init__(self):
        self.files = (Path().cwd() / "./data/StabGurevych17").glob('*.ann')
        
        self.claims = {}
        self.non_claims = {}
        
        i = 1
        w = 1

        for file in self.files:

            entities, relations, attributes, groups = \
                get_entities_relations_attributes_groups(file)
            
            for entity in entities.values():
                if entity.type == "MajorClaim" or entity.type == "Claim":
                    self.claims.update({i: entity.text})
                    i += 1
                else:
                    if entity.type == "Premise":
                        self.non_claims.update({w: entity.text})
                        w += 1

        # Convert dictionaries to data frames and shuffle them.

        cols = ["text"]

        self.df_claims = pd.DataFrame.from_dict(self.claims,
                                                orient='index',
                                                columns=cols)
        self.df_claims["target"] = 1

        self.df_non_claims = pd.DataFrame.from_dict(self.non_claims,
                                                    orient='index',
                                                    columns=cols)
        self.df_non_claims["target"] = 0

        # downsample majority class

        self.df_non_claims = self.df_non_claims.sample(n=len(
            self.df_claims), random_state=44)

        # shuffle rows with seed = 1

        self.df_all = pd.concat([self.df_claims, self.df_non_claims],
                                ignore_index=True)
        self.df_all = self.df_all.drop_duplicates().reset_index(drop=True)
        self.df_all = self.df_all.sample(frac=1, random_state=1).reset_index(drop=True)


class DaxenbergerModified:
    """Load corpus files to pandas frame"""

    def __init__(self):
        self.files = (Path().cwd() / "./data/Daxenberger17/PE").glob('*.csv')

        li = []

        for file in self.files:
            df = pd.read_csv(file, sep='\t', header=None)
            li.append(df)
        
        self.df_all = pd.concat(li, axis=0, ignore_index=True)
        self.df_all.columns = ["text", "target"]

        self.df_all = self.df_all.drop_duplicates().reset_index(drop=True)
        
        # downsample majority class

        print("Length of minority and majority \n")
        self.df_minority  = self.df_all[self.df_all['target']==1] # claims
        self.df_majority = self.df_all[self.df_all['target']==0] # non-claims
        self.df_majority = self.df_majority.sample(len(self.df_minority),
                                                   random_state=0)
        self.df_all = pd.concat([self.df_majority, self.df_minority])

        # shuffle rows with seed = 1

        self.df_all = self.df_all.sample(frac=1, random_state=1).reset_index(drop=True)


if __name__ == "__main__":

    # Load first corpus

    corpus_1 = StabGurevychCorpus()

    print(corpus_1.df_claims.shape)
    print(corpus_1.df_non_claims.shape)
    print(corpus_1.df_all.shape)
    print(corpus_1.df_all.head)
    
    print(f"Claims in corpus according to Stab and Gurevych (2017) 2257. Claims loaded: {len(corpus_1.claims)}")
    print(f"Non-claims in corpus according to Stab and Gurevych (2017) 3832. Non-claims loaded: {len(corpus_1.non_claims)}")

    baseline = len(corpus_1.non_claims) / (len(corpus_1.non_claims) + len(corpus_1.claims))

    print(f"Majority baseline: {round(baseline, 2)}")

    # Manual check of mean statement length from corpus
    sum_claim_n_clause = 0 
    sum_non_claim_n_clause = 0 

    for expression in corpus_1.claims.values():
        n = len(expression.split())
        sum_claim_n_clause += n
    
    print(f"Average length of claim at clause-level: {(sum_claim_n_clause / len(corpus_1.claims))}")

    for expression in corpus_1.non_claims.values():
        n = len(expression.split())
        sum_non_claim_n_clause += n
    
    print(f"Average length of non-claim at clause-level: {(sum_non_claim_n_clause / len(corpus_1.non_claims))}")

    ##################################################

    # Load second corpus

    corpus_2 = DaxenbergerModified()
    print(corpus_2.df_all.head)

    # Manual check of mean statement length from corpus
    sum_claim_n_sent = 0 
    sum_non_claim_n_sent = 0 

    for expression in corpus_2.df_minority["text"]:
        n = len(expression.split())
        sum_claim_n_sent += n
    
    print(f"Average length of claim at sentence level: {(sum_claim_n_sent / len(corpus_2.df_minority))}")

    for expression in corpus_2.df_majority["text"]:
        n = len(expression.split())
        sum_non_claim_n_sent += n
    
    print(f"Average length of non-claim at sentence level: {(sum_non_claim_n_sent / len(corpus_2.df_majority))}")