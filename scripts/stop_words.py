"""
This script creates a custom set of stop words. It is an adaption of SpaCy's 
stop word list, taken from SpaCy's stopword.py that can be downloaded
from https://github.com/explosion/spaCy/blob/master/spacy/lang/en/stop_words.py

To remove conjunctions and conjunctive adverbs from the stop word list,
I prompted ChatGPT (12.02.2024) to generate a list of the most frequent English 
conjunctions and conjunctive adverbs. It returned 82 words. 
"""

class StopWords:
    """
    The class is an adaption of SpaCy's stopword list, removing frequent 
    conjunctions and conjunctive adverbs.
    """

    def __init__(self):

        self.conjunctions = [
            "and", "but", "or", "so", "yet", "for", "nor", "because", "although",
            "since", "unless", "until", "whether", "while", "even though", "as",
            "after", "before", "though", "if", "provided", "once", "unless", "in case",
            "wherever", "wherever", "lest", "except", "insofar", "as long as",
            "as soon as", "whenever", "wherever", "rather than", "whether or not",
            "inasmuch as", "so that", "now that", "just as", "than", "as if",
            "as though", "while", "whereas", "regardless", "notwithstanding",
            "conversely", "alternatively", "however", "nevertheless", "nonetheless",
            "still", "instead", "yet", "on the other hand", "even so", "all the same",
            "in contrast", "on the contrary", "whereas"
        ]

        self.conjunctive_adverbs = [
            "accordingly", "also", "besides", "consequently", "finally", "furthermore",
            "hence", "however", "indeed", "instead", "likewise", "meanwhile", "moreover",
            "nevertheless", "nonetheless", "otherwise", "similarly", "still",
            "subsequently", "then", "therefore", "thus"
        ]

        self.ALL_CONJUNCTIONS = list(self.conjunctions + self.conjunctive_adverbs)

        self.STOP_WORDS = set(
        """
        a about above across after afterwards again against all almost alone along
        already also although always am among amongst amount an and another any anyhow
        anyone anything anyway anywhere are around as at

        back be became because become becomes becoming been before beforehand behind
        being below beside besides between beyond both bottom but by

        call can cannot ca could

        did do does doing done down due during

        each eight either eleven else elsewhere empty enough even ever every
        everyone everything everywhere except

        few fifteen fifty first five for former formerly forty four from front full
        further

        get give go

        had has have he hence her here hereafter hereby herein hereupon hers herself
        him himself his how however hundred

        i if in indeed into is it its itself

        keep

        last latter latterly least less

        just

        made make many may me meanwhile might mine more moreover most mostly move much
        must my myself

        name namely neither never nevertheless next nine no nobody none noone nor not
        nothing now nowhere

        of off often on once one only onto or other others otherwise our ours ourselves
        out over own

        part per perhaps please put

        quite

        rather re really regarding

        same say see seem seemed seeming seems serious several she should show side
        since six sixty so some somehow someone something sometime sometimes somewhere
        still such

        take ten than that the their them themselves then thence there thereafter
        thereby therefore therein thereupon these they third this those though three
        through throughout thru thus to together too top toward towards twelve twenty
        two

        under until up unless upon us used using

        various very very via was we well were what whatever when whence whenever where
        whereafter whereas whereby wherein whereupon wherever whether which while
        whither who whoever whole whom whose why will with within without would

        yet you your yours yourself yourselves
        """.split()
        )

        contractions = ["n't", "'d", "'ll", "'m", "'re", "'s", "'ve"]
        self.STOP_WORDS.update(contractions)

        for apostrophe in ["‘", "’"]:
            for stopword in contractions:
                self.STOP_WORDS.add(stopword.replace("'", apostrophe))
        
        self.STOP_WORDS = [word for word in self.STOP_WORDS if word not in self.ALL_CONJUNCTIONS]


if __name__ == "__main__":

    test_object = StopWords()
    
    print("All conjunctions only:\n")
    print(test_object.ALL_CONJUNCTIONS)
    print(len(test_object.ALL_CONJUNCTIONS))
    print("Stop words without conjunctions:\n")
    print(test_object.STOP_WORDS)
    print(len(test_object.STOP_WORDS))
    