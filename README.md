# terminology_evaluation

## Prerequisite
  1. **torch**
  2. **stanza**
  3. **argparse**
  4. **itertools**
  5. **sacrebleu**
  6. **transformers**
  7. **numpy**

## Arguments
  1. --language - The language code (eg. fr for French) of the targated language.
  2. --in_directory - This is a file contatining sentences of the targated language. Demo can be found at data/en-fr.hyp.txt.truecased.
  3. --unmodified_ref_directory - This is a file contatining reference sentences of the targated language. Demo can be found at data/unmodified_refs/all.fr.tsv.truecased.updated.
  4. --id_directory - This is a file which contains the ids of each sentence of the reference file. Demo can be found at data/ids/all.ids.txt.
  5. --modified_ref_directory - This is a file contatining reference sentences with additional information of the targated language. Demo can be found at data/modified_refs/exact.en-fr.all.updated.txt.
  6. --BLEU - True/False. By default False. If True shows that result.
  7. --EXACT_MATCH - True/False. By default False. If True shows that result.
  8. --TER - True/False. By default False. If True shows that result.
  9. --MOD_TER - True/False. By default False. If True shows that result.
  10. --ALIGN_EXACT - True/False. By default False. If True shows that result.
  11. --ALIGN_BLEU - True/False. By default False. If True shows that result.
  12. --ALIGN_UD - True/False. By default False. If True shows that result.
  13. --WINDOW_OVERLAP - True/False. By default False. If True shows that result.

## Example
    python3 evaluate_term_hyp.py 
    --language fr 
    --in_directory data/en-fr.hyp.txt.truecased 
    --unmodified_ref_directory data/unmodified_refs/all.fr.tsv.truecased.updated 
    --id_directory data/ids/all.ids.txt 
    --modified_ref_directory data/modified_refs/exact.en-fr.all.updated.txt 
    --BLEU True
