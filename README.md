# terminology_evaluation

## Installation and Prerequisites

The script uses Python 3. You can simply run the following to clone this repository and install all of the above requirements:

~~~
git clone https://github.com/mahfuzibnalam/terminology_evaluation.git
cd terminology_evaluation
pip install -r requirements.txt
~~~

List of requirements:
  1. **torch**
  2. **stanza**
  3. **argparse**
  4. **itertools**
  5. **sacrebleu**
  6. **transformers**
  7. **numpy**

## Code
The main script is `evaluate_term.py` that receives the following arguments:

  1. --language - The language code (eg. fr for French) of the target language.
  2. --hypothesis - This is the hypothesis file. Example file: `data/en-fr.hyp.txt.truecased`.
  3. --unmodified_ref_directory - This is a file with the references. An example file is provided at `data/unmodified_refs/all.fr.tsv.truecased.updated`.
  4. --id_directory - This is a file which contains the ids of each sentence of the reference file. Example: `data/ids/all.ids.txt`.
  5. --modified_ref_directory - This is a file containing target references with additional term information. Example file: `data/modified_refs/exact.en-fr.all.updated.txt`.
  6. --BLEU [True/False]. By default True. If True shows BLEU score.
  7. --EXACT_MATCH [True/False]. By default True. If True shows Exact Match score.
  8. --MOD_TER [True/False]. By default True. If True shows TERm score.
  9. --WINDOW_OVERLAP [True/False]. By default True. If True shows Window Overlap Score.
  10. --TER [True/False]. By default False. If True shows TER score.
  11. --ALIGN_EXACT [True/False]. By default False. If True shows that result.
  12. --ALIGN_BLEU [True/False]. By default False. If True shows that result.
  13. --ALIGN_UD [True/False]. By default False. If True shows that result.
  

## Example
You can test that your metrics work by running the following command on the sample data we provide.
~~~
python3 evaluate_term.py \
    --language fr  \
    --in_directory data/en-fr.hyp.txt.truecased \
    --unmodified_ref_directory data/unmodified_refs/all.fr.tsv.truecased.updated \
    --id_directory data/ids/all.ids.txt  \
    --modified_ref_directory data/modified_refs/exact.en-fr.all.updated.txt \
~~~
Running the above command will:
* Download the French Stanza models, if they are not available locally already
* Compute four metrics and print the following:
~~~
BLEU score: 46.097223339855816
Exact-Match Statistics
  Total correct: 2607
  Total wrong: 903
  Total correct (lemma): 340
  Total wrong (lemma): 66
Exact-Match Accuracy: 0.7525536261491318
Window Overlap Accuracy :
  Window 2:
  Exact Window Overlap Accuracy: 0.5932386427306287
  Window 3:
  Exact Window Overlap Accuracy: 0.5759241927272842

~~~

Notes: 
* The computation of TER or TERm can take quite some time if your data has very long sentences.
