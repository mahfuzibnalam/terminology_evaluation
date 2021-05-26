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
The main script is `evaluate_term_wmt.py` that receives the following arguments:

  1. --language - The language code (eg. fr for French) of the target language.
  2. --hypothesis - This is the hypothesis file. Example file: `data/en-fr.hyp.txt.truecased.xml`.
  3. --source_ref_directory - This is a file with the source references. An example file is provided at `data/all.en-fr.en.xml`
  4. --target_ref_directory - This is a file with the target references. An example file is provided at `data/all.en-fr.fr.xml`
  5. --BLEU [True/False]. By default True. If True shows BLEU score.
  6. --EXACT_MATCH [True/False]. By default True. If True shows Exact Match score.
  7. --MOD_TER [True/False]. By default True. If True shows TERm score.
  8. --WINDOW_OVERLAP [True/False]. By default True. If True shows Window Overlap Score.
  9. --TER [True/False]. By default False. If True shows TER score.
  10. --ALIGN_EXACT [True/False]. By default False. If True shows that result.
  11. --ALIGN_BLEU [True/False]. By default False. If True shows that result.
  12. --ALIGN_UD [True/False]. By default False. If True shows that result.
  

## Example
You can test that your metrics work by running the following command on the sample data we provide.
~~~
python3 evaluate_term_wmt.py \
    --language fr \
    --hypothesis data/en-fr.hyp.txt.truecased.xml \
    --source_ref_directory data/all.en-fr.en.xml \
    --target_ref_directory data/all.en-fr.fr.xml \
~~~
Running the above command will:
* Download the French Stanza models, if they are not available locally already
* Compute four metrics and print the following:
~~~
BLEU score: 46.097223339855816
Exact-Match Statistics
        Total correct: 2386
        Total wrong: 704
        Total correct (lemma): 0
        Total wrong (lemma): 234
Exact-Match Accuracy: 0.7178098676293622
Window Overlap Accuracy :
        Window 2:
        Exact Window Overlap Accuracy: 0.3068392941133903
        Window 3:
        Exact Window Overlap Accuracy: 0.29947678745871537
1 - TERm Score: 0.5981857378665372

~~~

Notes: 
* The computation of TER or TERm can take quite some time if your data has very long sentences.
