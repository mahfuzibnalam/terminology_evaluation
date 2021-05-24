import re
import TER
import torch
import stanza
import string
import argparse
import itertools
import sacrebleu
import numpy as np
import TER_modified
import transformers
import matplotlib.pyplot as plt
from collections import defaultdict
plt.style.use('seaborn-whitegrid')


def read_ids(f):
	with open(f) as inp:
		lines = inp.readlines()
	return [l.strip() for l in lines]

def read_reference_data(l):
	with open(l) as inp:
		lines = inp.readlines()
	ref = {}
	terms = []
	terms_l = []
	ID = None
	for l in lines:
		if (l == "\n"):
			if ID and (terms or terms_l) :
				ref[ID] = (source, reference, terms, terms_l)
				terms = []
				terms_l = []
				source = ''
				reference = ''
		elif (l[:4] == "SID:"):
			ID = l[5:].strip()
		elif (l[:2] == "S:"):
			source = " " + l[3:].strip() + " "
		elif (l[:2] == "T:"):
			reference = " " + l[3:].strip() + " "
		elif (l[:25] == "Term (Source and Target):"):
			terms.append(l[26:].strip())
		elif (l[:31] == "Term (Source Lemma and Target):"):
			terms.append(l[32:].strip())
		elif (l[:31] == "Term (Source and Target Lemma):"):
			terms_l.append(l[32:].strip())
		elif (l[:37] == "Term (Source Lemma and Target Lemma):"):
			terms_l.append(l[38:].strip())
	return ref

def read_reference_data_ENG(l):
	with open(l) as inp:
		lines = inp.readlines()
	ref = {}
	terms = []
	terms_l = []
	ID = None
	for l in lines:
		if (l == "\n"):
			if ID and (terms or terms_l) :
				ref[ID] = (source, reference, terms, terms_l)
				terms = []
				terms_l = []
				source = ''
				reference = ''
		elif (l[:4] == "SID:"):
			ID = l[5:].strip()
		elif (l[:2] == "S:"):
			source = " " + l[3:].strip() + " "
		elif (l[:2] == "T:"):
			reference = " " + l[3:].strip() + " "
		elif (l[:25] == "Term (Source and Target):"):
			terms.append(l[26:].strip())
		elif (l[:31] == "Term (Source and Target Lemma):"):
			terms.append(l[32:].strip())
		elif (l[:31] == "Term (Source Lemma and Target):"):
			terms_l.append(l[32:].strip())
		elif (l[:37] == "Term (Source Lemma and Target Lemma):"):
			terms_l.append(l[38:].strip())
	return ref

def read_outputs(f):
	with open(f) as inp:
		lines = inp.readlines()
	return [' ' + ' '.join(l.strip().split()) + ' ' for l in lines]



def compare_EXACT(hyp, ref):
	source, reference, terms, terms_l = ref
	count_correct = 0
	count_wrong = 0
	count_correct_l = 0
	count_wrong_l = 0
	desireds = defaultdict(int)
	desireds_l = defaultdict(int)
	for t in terms:
		t = t.split(' --> ')
		t = t[1].split(' ||| ')
		desired = "(?= " + t[0] + " )"
		desireds[desired] += 1
	for desired in desireds:
		desired_starts = [m.start() for m in re.finditer(desired, hyp)]
		cnt = len(desired_starts)
		if cnt < desireds[desired]:
			count_correct += cnt
			count_wrong += desireds[desired] - cnt
		else:
			count_correct += desireds[desired]
	if terms_l and SUPPORTED:
		doc_f = l2_stanza(hyp)
		hyp_l = ' ' + ' '.join([w.lemma for w in doc_f.sentences[0].words]) + ' '
		for t in terms_l:
			t = t.split(' --> ')
			t = t[1].split(' ||| ')
			desired_l = "(?= " + t[0] + " )"
			desireds_l[desired_l] += 1
		for desired_l in desireds_l:
			desired_l_starts = [m.start() for m in re.finditer(desired_l, hyp_l)]
			cnt = len(desired_l_starts)
			if cnt < desireds_l[desired_l]:
				count_correct_l += cnt
				count_wrong_l += desireds_l[desired_l] - cnt
			else:
				count_correct_l += desireds_l[desired_l]

	return count_correct, count_wrong, count_correct_l, count_wrong_l

def compare_EXACT_ENG(hyp, ref):
	source, reference, terms, terms_l = ref
	count_correct = 0
	count_wrong = 0
	count_correct_l = 0
	count_wrong_l = 0
	desireds = defaultdict(int)
	desireds_l = defaultdict(int)
	for t in terms:
		t = t.split(' --> ')
		t = t[0].split(' ||| ')
		desired = "(?= " + t[0] + " )"
		desireds[desired] += 1
	for desired in desireds:
		desired_starts = [m.start() for m in re.finditer(desired, hyp)]
		cnt = len(desired_starts)
		if cnt < desireds[desired]:
			count_correct += cnt
			count_wrong += desireds[desired] - cnt
		else:
			count_correct += desireds[desired]
	if terms_l and SUPPORTED:
		doc_f = l2_stanza(hyp)
		hyp_l = ' ' + ' '.join([w.lemma for w in doc_f.sentences[0].words]) + ' '
		for t in terms_l:
			t = t.split(' --> ')
			t = t[0].split(' ||| ')
			desired_l = "(?= " + t[0] + " )"
			desireds_l[desired_l] += 1
		for desired_l in desireds_l:
			desired_l_starts = [m.start() for m in re.finditer(desired_l, hyp_l)]
			cnt = len(desired_l_starts)
			if cnt < desireds_l[desired_l]:
				count_correct_l += cnt
				count_wrong_l += desireds_l[desired_l] - cnt
			else:
				count_correct_l += desireds_l[desired_l]
				
	return count_correct, count_wrong, count_correct_l, count_wrong_l

def compare_TER_w(hyp, ref, lc):
	source, reference, terms, terms_l = ref
	ter = 0
	term_ids = []
	for t in terms:
		t = t.split(' --> ')
		t = t[1].split(' ||| ')
		term_ids.extend(t[1].split(','))
	ter += TER_modified.ter(hyp.split(), reference.split(), lc, term_ids)
	term_l_ids = []
	if terms_l and SUPPORTED:
		doc_f = l2_stanza(hyp)
		hyp_l = ' ' + ' '.join([w.lemma for w in doc_f.sentences[0].words]) + ' '
		doc_f = l2_stanza(reference)
		reference_l = ' ' + ' '.join([w.lemma for w in doc_f.sentences[0].words]) + ' '
		for t in terms_l:
			t = t.split(' --> ')
			t = t[1].split(' ||| ')
			term_l_ids.extend(t[1].split(','))
		ter += TER_modified.ter(hyp_l.split(), reference_l.split(), lc, term_l_ids)
		ter /= 2.0
					
	return ter

def compare_TER_w_ENG(hyp, ref, lc):
	source, reference, terms, terms_l = ref
	ter = 0
	term_ids = []
	for t in terms:
		t = t.split(' --> ')
		t = t[1].split(' ||| ')
		term_ids.extend(t[1].split(','))
	ter += TER_modified.ter(hyp.split(), source.split(), lc, term_ids)
	term_l_ids = []
	if terms_l and SUPPORTED:
		doc_f = l2_stanza(hyp)
		hyp_l = ' ' + ' '.join([w.lemma for w in doc_f.sentences[0].words]) + ' '
		doc_f = l2_stanza(source)
		source_l = ' ' + ' '.join([w.lemma for w in doc_f.sentences[0].words]) + ' '
		for t in terms_l:
			t = t.split(' --> ')
			t = t[1].split(' ||| ')
			term_l_ids.extend(t[1].split(','))
		ter += TER_modified.ter(hyp_l.split(), source_l.split(), lc, term_l_ids)
		ter /= 2.0
					
	return ter

def compare_exact_align_bleu(hyp, ref, window):
	source, reference, terms, terms_l = ref
	bleu_score = 0.0
	matched = 0

	hyp_tokens = hyp.strip().split()
	reference_tokens = reference.strip().split()
	align_words = AWESOME(reference_tokens, hyp_tokens)
	desireds = {}
	for t in terms:
		t = t.split(' --> ')
		t = t[1].split(' ||| ')
		desired  = " " + t[0] + " "
		desiredindxs = [(int)(item)for item in t[1].split(",")]
		if desired in desireds:
			desireds[desired].append(desiredindxs)
		else:
			desireds[desired] = [desiredindxs]
	for desired in desireds:
		fts = [m.start() for m in re.finditer(f"(?={desired})", hyp)]
		mapped_target = []
		for listfs in desireds[desired]:
			ref_start_index = min(listfs) - window if min(listfs) - window >= 0 else 0
			ref_end_index = max(listfs) + window + 1 if max(listfs) + window < len(reference_tokens) else len(reference_tokens)
			windowed_ref = ' '.join(reference_tokens[ref_start_index : ref_end_index])
			alignfs = []
			for elfs in listfs:
				if elfs in align_words:
					alignfs.extend(align_words[elfs])
			for k, ft in enumerate(fts):
				if k not in mapped_target:
					cntfts = hyp.count(" ", 0, ft)
					listft = [item for item in range(cntfts, cntfts + len(desired.strip().split()))]
					match = True
					for elft in listft:
						if elft not in alignfs:
							match = False
							break
					if match:
						matched += 1
						mapped_target.append(k)
						hyp_start_index = min(listft) - window if min(listft) - window >= 0 else 0
						hyp_end_index = max(listft) + window + 1 if max(listft) + window < len(hyp_tokens) else len(hyp_tokens)
						windowed_hyp = ' '.join(hyp_tokens[hyp_start_index : hyp_end_index])
						bleu_score += sacrebleu.corpus_bleu([windowed_hyp], [[windowed_ref]]).score
						break
	
	if SUPPORTED and terms_l:
		doc_f = l2_stanza(hyp)
		hyp = ' ' + ' '.join([w.lemma for w in doc_f.sentences[0].words]) + ' '
		doc_f = l2_stanza(reference)
		reference = ' ' + ' '.join([w.lemma for w in doc_f.sentences[0].words]) + ' '
		hyp_tokens = hyp.strip().split()
		reference_tokens = reference.strip().split()
		align_words = AWESOME(reference_tokens, hyp_tokens)
		desireds = {}
		for t in terms_l:
			t = t.split(' --> ')
			t = t[1].split(' ||| ')
			desired  = " " + t[0] + " "
			desiredindxs = [(int)(item)for item in t[1].split(",")]
			if desired in desireds:
				desireds[desired].append(desiredindxs)
			else:
				desireds[desired] = [desiredindxs]
		for desired in desireds:
			fts = [m.start() for m in re.finditer(f"(?={desired})", hyp)]
			mapped_target = []
			for listfs in desireds[desired]:
				ref_start_index = min(listfs) - window if min(listfs) - window >= 0 else 0
				ref_end_index = max(listfs) + window + 1 if max(listfs) + window < len(reference_tokens) else len(reference_tokens)
				windowed_ref = ' '.join(reference_tokens[ref_start_index : ref_end_index])
				alignfs = []
				for elfs in listfs:
					if elfs in align_words:
						alignfs.extend(align_words[elfs])
				for k, ft in enumerate(fts):
					if k not in mapped_target:
						cntfts = hyp.count(" ", 0, ft)
						listft = [item for item in range(cntfts, cntfts + len(desired.strip().split()))]
						match = True
						for elft in listft:
							if elft not in alignfs:
								match = False
								break
						if match:
							matched += 1
							mapped_target.append(k)
							hyp_start_index = min(listft) - window if min(listft) - window >= 0 else 0
							hyp_end_index = max(listft) + window + 1 if max(listft) + window < len(hyp_tokens) else len(hyp_tokens)
							windowed_hyp = ' '.join(hyp_tokens[hyp_start_index : hyp_end_index])
							bleu_score += sacrebleu.corpus_bleu([windowed_hyp], [[windowed_ref]]).score
							break
	
	return bleu_score / matched if matched > 0 else 0

def compare_exact_align_bleu_ENG(hyp, ref, window):
	reference, source, terms, terms_l = ref
	bleu_score = 0.0
	matched = 0

	hyp_tokens = hyp.strip().split()
	reference_tokens = reference.strip().split()
	align_words = AWESOME(reference_tokens, hyp_tokens)
	desireds = {}
	for t in terms:
		t = t.split(' --> ')
		t = t[0].split(' ||| ')
		desired  = " " + t[0] + " "
		desiredindxs = [(int)(item)for item in t[1].split(",")]
		if desired in desireds:
			desireds[desired].append(desiredindxs)
		else:
			desireds[desired] = [desiredindxs]
	for desired in desireds:
		fts = [m.start() for m in re.finditer(f"(?={desired})", hyp)]
		mapped_target = []
		for listfs in desireds[desired]:
			ref_start_index = min(listfs) - window if min(listfs) - window >= 0 else 0
			ref_end_index = max(listfs) + window + 1 if max(listfs) + window < len(reference_tokens) else len(reference_tokens)
			windowed_ref = ' '.join(reference_tokens[ref_start_index : ref_end_index])
			alignfs = []
			for elfs in listfs:
				if elfs in align_words:
					alignfs.extend(align_words[elfs])
			for k, ft in enumerate(fts):
				if k not in mapped_target:
					cntfts = hyp.count(" ", 0, ft)
					listft = [item for item in range(cntfts, cntfts + len(desired.strip().split()))]
					match = True
					for elft in listft:
						if elft not in alignfs:
							match = False
							break
					if match:
						matched += 1
						mapped_target.append(k)
						hyp_start_index = min(listft) - window if min(listft) - window >= 0 else 0
						hyp_end_index = max(listft) + window + 1 if max(listft) + window < len(hyp_tokens) else len(hyp_tokens)
						windowed_hyp = ' '.join(hyp_tokens[hyp_start_index : hyp_end_index])
						bleu_score += sacrebleu.corpus_bleu([windowed_hyp], [[windowed_ref]]).score
						break
	
	if SUPPORTED and terms_l:
		doc_f = l2_stanza(hyp)
		hyp = ' ' + ' '.join([w.lemma for w in doc_f.sentences[0].words]) + ' '
		doc_f = l2_stanza(reference)
		reference = ' ' + ' '.join([w.lemma for w in doc_f.sentences[0].words]) + ' '
		hyp_tokens = hyp.strip().split()
		reference_tokens = reference.strip().split()
		align_words = AWESOME(reference_tokens, hyp_tokens)
		desireds = {}
		for t in terms_l:
			t = t.split(' --> ')
			t = t[0].split(' ||| ')
			desired  = " " + t[0] + " "
			desiredindxs = [(int)(item)for item in t[1].split(",")]
			if desired in desireds:
				desireds[desired].append(desiredindxs)
			else:
				desireds[desired] = [desiredindxs]
		for desired in desireds:
			fts = [m.start() for m in re.finditer(f"(?={desired})", hyp)]
			mapped_target = []
			for listfs in desireds[desired]:
				ref_start_index = min(listfs) - window if min(listfs) - window >= 0 else 0
				ref_end_index = max(listfs) + window + 1 if max(listfs) + window < len(reference_tokens) else len(reference_tokens)
				windowed_ref = ' '.join(reference_tokens[ref_start_index : ref_end_index])
				alignfs = []
				for elfs in listfs:
					if elfs in align_words:
						alignfs.extend(align_words[elfs])
				for k, ft in enumerate(fts):
					if k not in mapped_target:
						cntfts = hyp.count(" ", 0, ft)
						listft = [item for item in range(cntfts, cntfts + len(desired.strip().split()))]
						match = True
						for elft in listft:
							if elft not in alignfs:
								match = False
								break
						if match:
							matched += 1
							mapped_target.append(k)
							hyp_start_index = min(listft) - window if min(listft) - window >= 0 else 0
							hyp_end_index = max(listft) + window + 1 if max(listft) + window < len(hyp_tokens) else len(hyp_tokens)
							windowed_hyp = ' '.join(hyp_tokens[hyp_start_index : hyp_end_index])
							bleu_score += sacrebleu.corpus_bleu([windowed_hyp], [[windowed_ref]]).score
							break
	
	return bleu_score / matched if matched > 0 else 0

def compare_exact_align(hyp, ref, window):
	source, reference, terms, terms_l = ref
	acc = 0.0
	matched = 0

	hyp_tokens = hyp.strip().split()
	reference_tokens = reference.strip().split()
	align_words = AWESOME(reference_tokens, hyp_tokens)
	desireds = {}
	for t in terms:
		t = t.split(' --> ')
		t = t[1].split(' ||| ')
		desired  = " " + t[0] + " "
		desiredindxs = [(int)(item)for item in t[1].split(",")]
		if desired in desireds:
			desireds[desired].append(desiredindxs)
		else:
			desireds[desired] = [desiredindxs]
	for desired in desireds:
		fts = [m.start() for m in re.finditer(f"(?={desired})", hyp)]
		mapped_target = []
		for listfs in desireds[desired]:
			ref_start_index = min(listfs) - window if min(listfs) - window >= 0 else 0
			ref_end_index = max(listfs) + window + 1 if max(listfs) + window < len(reference_tokens) else len(reference_tokens)
			alignfs = []
			for elfs in listfs:
				if elfs in align_words:
					alignfs.extend(align_words[elfs])
			for k, ft in enumerate(fts):
				if k not in mapped_target:
					cntfts = hyp.count(" ", 0, ft)
					listft = [item for item in range(cntfts, cntfts + len(desired.strip().split()))]
					match = True
					for elft in listft:
						if elft not in alignfs:
							match = False
							break
					if match:
						matched += 1
						mapped_target.append(k)
						cnt = 0
						for win in range(1, window + 1):
							if min(listfs) - win >= 0 and min(listft) - win >= 0:
								if reference_tokens[min(listfs) - win] == hyp_tokens[min(listft) - win]:
									cnt += 1
						for win in range(1, window + 1):
							if max(listfs) + win < len(reference_tokens) and max(listft) + win < len(hyp_tokens):
								if reference_tokens[max(listfs) + win] == hyp_tokens[max(listft) + win]:
									cnt += 1
						acc += cnt / (ref_end_index - ref_start_index - len(listfs))
						break
	
	if SUPPORTED and terms_l:
		doc_f = l2_stanza(hyp)
		hyp = ' ' + ' '.join([w.lemma for w in doc_f.sentences[0].words]) + ' '
		doc_f = l2_stanza(reference)
		reference = ' ' + ' '.join([w.lemma for w in doc_f.sentences[0].words]) + ' '
		hyp_tokens = hyp.strip().split()
		reference_tokens = reference.strip().split()
		align_words = AWESOME(reference_tokens, hyp_tokens)
		desireds = {}
		for t in terms_l:
			t = t.split(' --> ')
			t = t[1].split(' ||| ')
			desired  = " " + t[0] + " "
			desiredindxs = [(int)(item)for item in t[1].split(",")]
			if desired in desireds:
				desireds[desired].append(desiredindxs)
			else:
				desireds[desired] = [desiredindxs]
		for desired in desireds:
			fts = [m.start() for m in re.finditer(f"(?={desired})", hyp)]
			mapped_target = []
			for listfs in desireds[desired]:
				ref_start_index = min(listfs) - window if min(listfs) - window >= 0 else 0
				ref_end_index = max(listfs) + window + 1 if max(listfs) + window < len(reference_tokens) else len(reference_tokens)
				alignfs = []
				for elfs in listfs:
					if elfs in align_words:
						alignfs.extend(align_words[elfs])
				for k, ft in enumerate(fts):
					if k not in mapped_target:
						cntfts = hyp.count(" ", 0, ft)
						listft = [item for item in range(cntfts, cntfts + len(desired.strip().split()))]
						match = True
						for elft in listft:
							if elft not in alignfs:
								match = False
								break
						if match:
							matched += 1
							mapped_target.append(k)
							cnt = 0
							for win in range(1, window + 1):
								if min(listfs) - win >= 0 and min(listft) - win >= 0:
									if reference_tokens[min(listfs) - win] == hyp_tokens[min(listft) - win]:
										cnt += 1
							for win in range(1, window + 1):
								if max(listfs) + win < len(reference_tokens) and max(listft) + win < len(hyp_tokens):
									if reference_tokens[max(listfs) + win] == hyp_tokens[max(listft) + win]:
										cnt += 1
							acc += cnt / (ref_end_index - ref_start_index - len(listfs))
							break
	
	return acc / matched if matched > 0 else 0

def compare_exact_align_ENG(hyp, ref, window):
	reference, source, terms, terms_l = ref
	acc = 0.0
	matched = 0

	hyp_tokens = hyp.strip().split()
	reference_tokens = reference.strip().split()
	align_words = AWESOME(reference_tokens, hyp_tokens)
	desireds = {}
	for t in terms:
		t = t.split(' --> ')
		t = t[0].split(' ||| ')
		desired  = " " + t[0] + " "
		desiredindxs = [(int)(item)for item in t[1].split(",")]
		if desired in desireds:
			desireds[desired].append(desiredindxs)
		else:
			desireds[desired] = [desiredindxs]
	for desired in desireds:
		fts = [m.start() for m in re.finditer(f"(?={desired})", hyp)]
		mapped_target = []
		for listfs in desireds[desired]:
			ref_start_index = min(listfs) - window if min(listfs) - window >= 0 else 0
			ref_end_index = max(listfs) + window + 1 if max(listfs) + window < len(reference_tokens) else len(reference_tokens)
			alignfs = []
			for elfs in listfs:
				if elfs in align_words:
					alignfs.extend(align_words[elfs])
			for k, ft in enumerate(fts):
				if k not in mapped_target:
					cntfts = hyp.count(" ", 0, ft)
					listft = [item for item in range(cntfts, cntfts + len(desired.strip().split()))]
					match = True
					for elft in listft:
						if elft not in alignfs:
							match = False
							break
					if match:
						matched += 1
						mapped_target.append(k)
						cnt = 0
						for win in range(1, window + 1):
							if min(listfs) - win >= 0 and min(listft) - win >= 0:
								if reference_tokens[min(listfs) - win] == hyp_tokens[min(listft) - win]:
									cnt += 1
						for win in range(1, window + 1):
							if max(listfs) + win < len(reference_tokens) and max(listft) + win < len(hyp_tokens):
								if reference_tokens[max(listfs) + win] == hyp_tokens[max(listft) + win]:
									cnt += 1
						acc += cnt / (ref_end_index - ref_start_index - len(listfs))
						break
	
	if SUPPORTED and terms_l:
		doc_f = l2_stanza(hyp)
		hyp = ' ' + ' '.join([w.lemma for w in doc_f.sentences[0].words]) + ' '
		doc_f = l2_stanza(reference)
		reference = ' ' + ' '.join([w.lemma for w in doc_f.sentences[0].words]) + ' '
		hyp_tokens = hyp.strip().split()
		reference_tokens = reference.strip().split()
		align_words = AWESOME(reference_tokens, hyp_tokens)
		desireds = {}
		for t in terms_l:
			t = t.split(' --> ')
			t = t[0].split(' ||| ')
			desired  = " " + t[0] + " "
			desiredindxs = [(int)(item)for item in t[1].split(",")]
			if desired in desireds:
				desireds[desired].append(desiredindxs)
			else:
				desireds[desired] = [desiredindxs]
		for desired in desireds:
			fts = [m.start() for m in re.finditer(f"(?={desired})", hyp)]
			mapped_target = []
			for listfs in desireds[desired]:
				ref_start_index = min(listfs) - window if min(listfs) - window >= 0 else 0
				ref_end_index = max(listfs) + window + 1 if max(listfs) + window < len(reference_tokens) else len(reference_tokens)
				alignfs = []
				for elfs in listfs:
					if elfs in align_words:
						alignfs.extend(align_words[elfs])
				for k, ft in enumerate(fts):
					if k not in mapped_target:
						cntfts = hyp.count(" ", 0, ft)
						listft = [item for item in range(cntfts, cntfts + len(desired.strip().split()))]
						match = True
						for elft in listft:
							if elft not in alignfs:
								match = False
								break
						if match:
							matched += 1
							mapped_target.append(k)
							cnt = 0
							for win in range(1, window + 1):
								if min(listfs) - win >= 0 and min(listft) - win >= 0:
									if reference_tokens[min(listfs) - win] == hyp_tokens[min(listft) - win]:
										cnt += 1
							for win in range(1, window + 1):
								if max(listfs) + win < len(reference_tokens) and max(listft) + win < len(hyp_tokens):
									if reference_tokens[max(listfs) + win] == hyp_tokens[max(listft) + win]:
										cnt += 1
							acc += cnt / (ref_end_index - ref_start_index - len(listfs))
							break
	
	return acc / matched if matched > 0 else 0

def compare_exact_window_overlap(hyp, ref, window):
	source, reference, terms, terms_l = ref
	accuracy = 0.0
	matched = 0

	hyp_tokens = hyp.strip().split()
	reference_tokens = reference.strip().split()
	desireds = {}
	for t in terms:
		t = t.split(' --> ')
		t = t[1].split(' ||| ')
		desired  = " " + t[0] + " "
		desiredindxs = [(int)(item)for item in t[1].split(",")]
		if desired in desireds:
			desireds[desired].append(desiredindxs)
		else:
			desireds[desired] = [desiredindxs]
	for desired in desireds:
		fts = [m.start() for m in re.finditer(f"(?={desired})", hyp)]
		accs = {}
		for j, listfs in enumerate(desireds[desired]):
			ref_words = []
			for win in range(min(listfs) - 1, -1, -1):
				if reference_tokens[win] not in string.punctuation:
					ref_words.append(reference_tokens[win])
					if len(ref_words) == window:
						break
			ref_wordsr = []
			for win in range(max(listfs) + 1, len(reference_tokens), 1):
				if reference_tokens[win] not in string.punctuation:
					ref_wordsr.append(reference_tokens[win])
					if len(ref_wordsr) == window:
						break
			ref_words.extend(ref_wordsr)
			for k, ft in enumerate(fts):
				cntfts = hyp.count(" ", 0, ft)
				listft = [item for item in range(cntfts, cntfts + len(desired.strip().split()))]
				hyp_words = []
				for win in range(min(listft) - 1, -1, -1):
					if hyp_tokens[win] not in string.punctuation:
						hyp_words.append(hyp_tokens[win])
						if len(hyp_words) == window:
							break
				hyp_wordsr = []
				for win in range(max(listft) + 1, len(hyp_tokens), 1):
					if hyp_tokens[win] not in string.punctuation:
						hyp_wordsr.append(hyp_tokens[win])
						if len(hyp_wordsr) == window:
							break
				hyp_words.extend(hyp_wordsr)
				cnt = 0
				for ref_word in ref_words:
					if ref_word in hyp_words:
						cnt += 1
						hyp_words.remove(ref_word)
				accs[f"{j}-{k}"] = cnt / len(ref_words) if len(ref_words) != 0 else + 0
		accs = dict(sorted(accs.items(), key=lambda item: item[1], reverse=True))
		mapped_ref = []
		mapped_hyp = []
		for acc in accs:
			if acc.split("-")[0] not in mapped_ref and acc.split("-")[1] not in mapped_hyp:
				mapped_ref.append(acc.split("-")[0])
				mapped_hyp.append(acc.split("-")[1])
				matched += 1
				accuracy += accs[acc]
	
	if SUPPORTED and terms_l:
		doc_f = l2_stanza(hyp)
		hyp = ' ' + ' '.join([w.lemma for w in doc_f.sentences[0].words]) + ' '
		doc_f = l2_stanza(reference)
		reference = ' ' + ' '.join([w.lemma for w in doc_f.sentences[0].words]) + ' '
		hyp_tokens = hyp.strip().split()
		reference_tokens = reference.strip().split()
		desireds = {}
		for t in terms_l:
			t = t.split(' --> ')
			t = t[1].split(' ||| ')
			desired  = " " + t[0] + " "
			desiredindxs = [(int)(item)for item in t[1].split(",")]
			if desired in desireds:
				desireds[desired].append(desiredindxs)
			else:
				desireds[desired] = [desiredindxs]
		for desired in desireds:
			fts = [m.start() for m in re.finditer(f"(?={desired})", hyp)]
			accs = {}
			for j, listfs in enumerate(desireds[desired]):
				ref_words = []
				for win in range(min(listfs) - 1, -1, -1):
					if reference_tokens[win] not in string.punctuation:
						ref_words.append(reference_tokens[win])
						if len(ref_words) == window:
							break
				ref_wordsr = []
				for win in range(max(listfs) + 1, len(reference_tokens), 1):
					if reference_tokens[win] not in string.punctuation:
						ref_wordsr.append(reference_tokens[win])
						if len(ref_wordsr) == window:
							break
				ref_words.extend(ref_wordsr)
				for k, ft in enumerate(fts):
					cntfts = hyp.count(" ", 0, ft)
					listft = [item for item in range(cntfts, cntfts + len(desired.strip().split()))]
					hyp_words = []
					for win in range(min(listft) - 1, -1, -1):
						if hyp_tokens[win] not in string.punctuation:
							hyp_words.append(hyp_tokens[win])
							if len(hyp_words) == window:
								break
					hyp_wordsr = []
					for win in range(max(listft) + 1, len(hyp_tokens), 1):
						if hyp_tokens[win] not in string.punctuation:
							hyp_wordsr.append(hyp_tokens[win])
							if len(hyp_wordsr) == window:
								break
					hyp_words.extend(hyp_wordsr)
					cnt = 0
					for ref_word in ref_words:
						if ref_word in hyp_words:
							cnt += 1
							hyp_words.remove(ref_word)
					accs[f"{j}-{k}"] = cnt / len(ref_words) if len(ref_words) != 0 else + 0
			accs = dict(sorted(accs.items(), key=lambda item: item[1], reverse=True))
			mapped_ref = []
			mapped_hyp = []
			for acc in accs:
				if acc.split("-")[0] not in mapped_ref and acc.split("-")[1] not in mapped_hyp:
					mapped_ref.append(acc.split("-")[0])
					mapped_hyp.append(acc.split("-")[1])
					matched += 1
					accuracy += accs[acc]
	
	return accuracy / matched if matched > 0 else 0

def compare_exact_window_overlap_ENG(hyp, ref, window):
	reference, source, terms, terms_l = ref
	accuracy = 0.0
	matched = 0

	hyp_tokens = hyp.strip().split()
	reference_tokens = reference.strip().split()
	desireds = {}
	for t in terms:
		t = t.split(' --> ')
		t = t[0].split(' ||| ')
		desired  = " " + t[0] + " "
		desiredindxs = [(int)(item)for item in t[1].split(",")]
		if desired in desireds:
			desireds[desired].append(desiredindxs)
		else:
			desireds[desired] = [desiredindxs]
	for desired in desireds:
		fts = [m.start() for m in re.finditer(f"(?={desired})", hyp)]
		accs = {}
		for j, listfs in enumerate(desireds[desired]):
			ref_words = []
			for win in range(min(listfs) - 1, -1, -1):
				if reference_tokens[win] not in string.punctuation:
					ref_words.append(reference_tokens[win])
					if len(ref_words) == window:
						break
			ref_wordsr = []
			for win in range(max(listfs) + 1, len(reference_tokens), 1):
				if reference_tokens[win] not in string.punctuation:
					ref_wordsr.append(reference_tokens[win])
					if len(ref_wordsr) == window:
						break
			ref_words.extend(ref_wordsr)
			for k, ft in enumerate(fts):
				cntfts = hyp.count(" ", 0, ft)
				listft = [item for item in range(cntfts, cntfts + len(desired.strip().split()))]
				hyp_words = []
				for win in range(min(listft) - 1, -1, -1):
					if hyp_tokens[win] not in string.punctuation:
						hyp_words.append(hyp_tokens[win])
						if len(hyp_words) == window:
							break
				hyp_wordsr = []
				for win in range(max(listft) + 1, len(hyp_tokens), 1):
					if hyp_tokens[win] not in string.punctuation:
						hyp_wordsr.append(hyp_tokens[win])
						if len(hyp_wordsr) == window:
							break
				hyp_words.extend(hyp_wordsr)
				cnt = 0
				for ref_word in ref_words:
					if ref_word in hyp_words:
						cnt += 1
						hyp_words.remove(ref_word)
				accs[f"{j}-{k}"] = cnt / len(ref_words) if len(ref_words) != 0 else + 0
		accs = dict(sorted(accs.items(), key=lambda item: item[1], reverse=True))
		mapped_ref = []
		mapped_hyp = []
		for acc in accs:
			if acc.split("-")[0] not in mapped_ref and acc.split("-")[1] not in mapped_hyp:
				mapped_ref.append(acc.split("-")[0])
				mapped_hyp.append(acc.split("-")[1])
				matched += 1
				accuracy += accs[acc]
	
	if SUPPORTED and terms_l:
		doc_f = l2_stanza(hyp)
		hyp = ' ' + ' '.join([w.lemma for w in doc_f.sentences[0].words]) + ' '
		doc_f = l2_stanza(reference)
		reference = ' ' + ' '.join([w.lemma for w in doc_f.sentences[0].words]) + ' '
		hyp_tokens = hyp.strip().split()
		reference_tokens = reference.strip().split()
		desireds = {}
		for t in terms_l:
			t = t.split(' --> ')
			t = t[0].split(' ||| ')
			desired  = " " + t[0] + " "
			desiredindxs = [(int)(item)for item in t[1].split(",")]
			if desired in desireds:
				desireds[desired].append(desiredindxs)
			else:
				desireds[desired] = [desiredindxs]
		for desired in desireds:
			fts = [m.start() for m in re.finditer(f"(?={desired})", hyp)]
			accs = {}
			for j, listfs in enumerate(desireds[desired]):
				ref_words = []
				for win in range(min(listfs) - 1, -1, -1):
					if reference_tokens[win] not in string.punctuation:
						ref_words.append(reference_tokens[win])
						if len(ref_words) == window:
							break
				ref_wordsr = []
				for win in range(max(listfs) + 1, len(reference_tokens), 1):
					if reference_tokens[win] not in string.punctuation:
						ref_wordsr.append(reference_tokens[win])
						if len(ref_wordsr) == window:
							break
				ref_words.extend(ref_wordsr)
				for k, ft in enumerate(fts):
					cntfts = hyp.count(" ", 0, ft)
					listft = [item for item in range(cntfts, cntfts + len(desired.strip().split()))]
					hyp_words = []
					for win in range(min(listft) - 1, -1, -1):
						if hyp_tokens[win] not in string.punctuation:
							hyp_words.append(hyp_tokens[win])
							if len(hyp_words) == window:
								break
					hyp_wordsr = []
					for win in range(max(listft) + 1, len(hyp_tokens), 1):
						if hyp_tokens[win] not in string.punctuation:
							hyp_wordsr.append(hyp_tokens[win])
							if len(hyp_wordsr) == window:
								break
					hyp_words.extend(hyp_wordsr)
					cnt = 0
					for ref_word in ref_words:
						if ref_word in hyp_words:
							cnt += 1
							hyp_words.remove(ref_word)
					accs[f"{j}-{k}"] = cnt / len(ref_words) if len(ref_words) != 0 else + 0
			accs = dict(sorted(accs.items(), key=lambda item: item[1], reverse=True))
			mapped_ref = []
			mapped_hyp = []
			for acc in accs:
				if acc.split("-")[0] not in mapped_ref and acc.split("-")[1] not in mapped_hyp:
					mapped_ref.append(acc.split("-")[0])
					mapped_hyp.append(acc.split("-")[1])
					matched += 1
					accuracy += accs[acc]
	
	return accuracy / matched if matched > 0 else 0

def compare_exact_align_UD(hyp, ref):
	source, reference, terms, terms_l = ref
	acc = 0.0
	matched = 0

	hyp_tokens = hyp.strip().split()
	reference_tokens = reference.strip().split()
	align_words = AWESOME(reference_tokens, hyp_tokens)
	desireds = {}
	for t in terms:
		t = t.split(' --> ')
		t = t[1].split(' ||| ')
		desired  = " " + t[0] + " "
		desiredindxs = [(int)(item)for item in t[1].split(",")]
		if desired in desireds:
			desireds[desired].append(desiredindxs)
		else:
			desireds[desired] = [desiredindxs]
	for desired in desireds:
		fts = [m.start() for m in re.finditer(f"(?={desired})", hyp)]
		mapped_target = []
		for listfs in desireds[desired]:
			alignfs = []
			for elfs in listfs:
				if elfs in align_words:
					alignfs.extend(align_words[elfs])
			for k, ft in enumerate(fts):
				if k not in mapped_target:
					cntfts = hyp.count(" ", 0, ft)
					listft = [item for item in range(cntfts, cntfts + len(desired.strip().split()))]
					match = True
					for elft in listft:
						if elft not in alignfs:
							match = False
							break
					if match:
						matched += 1
						mapped_target.append(k)
						ref_doc = l2_stanza(reference).sentences[0]
						hyp_doc = l2_stanza(hyp).sentences[0]
						numerator = 0
						denominator = 0
						for elid, elfs in enumerate(listfs):
							tree_ref = [[ref_doc.words[listfs[elid]].deprel, ref_doc.words[ref_doc.words[listfs[elid]].head - 1].text if ref_doc.words[listfs[elid]].head > 0 else "root"]]
							for word in ref_doc.words:
								if word.head == listfs[elid] + 1:
									tree_ref.append([word.deprel, word.text])
							tree_hyp = [[hyp_doc.words[listft[elid]].deprel, hyp_doc.words[hyp_doc.words[listft[elid]].head - 1].text if hyp_doc.words[listft[elid]].head > 0 else "root"]]
							for word in hyp_doc.words:
								if word.head == listft[elid] + 1:
									tree_hyp.append([word.deprel, word.text])
							
							denominator += len(tree_ref)
							for node_ref in tree_ref:
								edge = node_ref[0]
								node = node_ref[1]
								for node_hyp in tree_hyp:
									if edge == node_hyp[0] and node == node_hyp[1]:
										numerator+= 1
										break
						acc += (numerator / denominator) if denominator else 0
						break
	
	if SUPPORTED and terms_l:
		hyp_doc = l2_stanza(hyp).sentences[0]
		hyp = ' ' + ' '.join([w.lemma for w in hyp_doc.words]) + ' '
		ref_doc = l2_stanza(reference).sentences[0]
		reference = ' ' + ' '.join([w.lemma for w in ref_doc.words]) + ' '
		hyp_tokens = hyp.strip().split()
		reference_tokens = reference.strip().split()
		align_words = AWESOME(reference_tokens, hyp_tokens)
		desireds = {}
		for t in terms_l:
			t = t.split(' --> ')
			t = t[1].split(' ||| ')
			desired  = " " + t[0] + " "
			desiredindxs = [(int)(item)for item in t[1].split(",")]
			if desired in desireds:
				desireds[desired].append(desiredindxs)
			else:
				desireds[desired] = [desiredindxs]
		for desired in desireds:
			fts = [m.start() for m in re.finditer(f"(?={desired})", hyp)]
			mapped_target = []
			for listfs in desireds[desired]:
				alignfs = []
				for elfs in listfs:
					if elfs in align_words:
						alignfs.extend(align_words[elfs])
				for k, ft in enumerate(fts):
					if k not in mapped_target:
						cntfts = hyp.count(" ", 0, ft)
						listft = [item for item in range(cntfts, cntfts + len(desired.strip().split()))]
						match = True
						for elft in listft:
							if elft not in alignfs:
								match = False
								break
						if match:
							matched += 1
							mapped_target.append(k)
							numerator = 0
							denominator = 0
							for elid, elfs in enumerate(listfs):
								tree_ref = [[ref_doc.words[listfs[elid]].deprel, ref_doc.words[ref_doc.words[listfs[elid]].head - 1].text if ref_doc.words[listfs[elid]].head > 0 else "root"]]
								for word in ref_doc.words:
									if word.head == listfs[elid] + 1:
										tree_ref.append([word.deprel, word.text])
								tree_hyp = [[hyp_doc.words[listft[elid]].deprel, hyp_doc.words[hyp_doc.words[listft[elid]].head - 1].text if hyp_doc.words[listft[elid]].head > 0 else "root"]]
								for word in hyp_doc.words:
									if word.head == listft[elid] + 1:
										tree_hyp.append([word.deprel, word.text])
								
								denominator += len(tree_ref)
								for node_ref in tree_ref:
									edge = node_ref[0]
									node = node_ref[1]
									for node_hyp in tree_hyp:
										if edge == node_hyp[0] and node == node_hyp[1]:
											numerator+= 1
											break
							acc += (numerator / denominator) if denominator else 0
							break
	
	return acc / matched if matched > 0 else 0

def compare_exact_align_UD_ENG(hyp, ref):
	reference, source, terms, terms_l = ref
	acc = 0.0
	matched = 0

	hyp_tokens = hyp.strip().split()
	reference_tokens = reference.strip().split()
	align_words = AWESOME(reference_tokens, hyp_tokens)
	desireds = {}
	for t in terms:
		t = t.split(' --> ')
		t = t[0].split(' ||| ')
		desired  = " " + t[0] + " "
		desiredindxs = [(int)(item)for item in t[1].split(",")]
		if desired in desireds:
			desireds[desired].append(desiredindxs)
		else:
			desireds[desired] = [desiredindxs]
	for desired in desireds:
		fts = [m.start() for m in re.finditer(f"(?={desired})", hyp)]
		mapped_target = []
		for listfs in desireds[desired]:
			alignfs = []
			for elfs in listfs:
				if elfs in align_words:
					alignfs.extend(align_words[elfs])
			for k, ft in enumerate(fts):
				if k not in mapped_target:
					cntfts = hyp.count(" ", 0, ft)
					listft = [item for item in range(cntfts, cntfts + len(desired.strip().split()))]
					match = True
					for elft in listft:
						if elft not in alignfs:
							match = False
							break
					if match:
						matched += 1
						mapped_target.append(k)
						ref_doc = l2_stanza(reference).sentences[0]
						hyp_doc = l2_stanza(hyp).sentences[0]
						numerator = 0
						denominator = 0
						for elid, elfs in enumerate(listfs):
							tree_ref = [[ref_doc.words[listfs[elid]].deprel, ref_doc.words[ref_doc.words[listfs[elid]].head - 1].text if ref_doc.words[listfs[elid]].head > 0 else "root"]]
							for word in ref_doc.words:
								if word.head == listfs[elid] + 1:
									tree_ref.append([word.deprel, word.text])
							tree_hyp = [[hyp_doc.words[listft[elid]].deprel, hyp_doc.words[hyp_doc.words[listft[elid]].head - 1].text if hyp_doc.words[listft[elid]].head > 0 else "root"]]
							for word in hyp_doc.words:
								if word.head == listft[elid] + 1:
									tree_hyp.append([word.deprel, word.text])
							
							denominator += len(tree_ref)
							for node_ref in tree_ref:
								edge = node_ref[0]
								node = node_ref[1]
								for node_hyp in tree_hyp:
									if edge == node_hyp[0] and node == node_hyp[1]:
										numerator+= 1
										break
						acc += (numerator / denominator) if denominator else 0
						break
	
	if SUPPORTED and terms_l:
		hyp_doc = l2_stanza(hyp).sentences[0]
		hyp = ' ' + ' '.join([w.lemma for w in hyp_doc.words]) + ' '
		ref_doc = l2_stanza(reference).sentences[0]
		reference = ' ' + ' '.join([w.lemma for w in ref_doc.words]) + ' '
		hyp_tokens = hyp.strip().split()
		reference_tokens = reference.strip().split()
		align_words = AWESOME(reference_tokens, hyp_tokens)
		desireds = {}
		for t in terms_l:
			t = t.split(' --> ')
			t = t[0].split(' ||| ')
			desired  = " " + t[0] + " "
			desiredindxs = [(int)(item)for item in t[1].split(",")]
			if desired in desireds:
				desireds[desired].append(desiredindxs)
			else:
				desireds[desired] = [desiredindxs]
		for desired in desireds:
			fts = [m.start() for m in re.finditer(f"(?={desired})", hyp)]
			mapped_target = []
			for listfs in desireds[desired]:
				alignfs = []
				for elfs in listfs:
					if elfs in align_words:
						alignfs.extend(align_words[elfs])
				for k, ft in enumerate(fts):
					if k not in mapped_target:
						cntfts = hyp.count(" ", 0, ft)
						listft = [item for item in range(cntfts, cntfts + len(desired.strip().split()))]
						match = True
						for elft in listft:
							if elft not in alignfs:
								match = False
								break
						if match:
							matched += 1
							mapped_target.append(k)
							numerator = 0
							denominator = 0
							for elid, elfs in enumerate(listfs):
								tree_ref = [[ref_doc.words[listfs[elid]].deprel, ref_doc.words[ref_doc.words[listfs[elid]].head - 1].text if ref_doc.words[listfs[elid]].head > 0 else "root"]]
								for word in ref_doc.words:
									if word.head == listfs[elid] + 1:
										tree_ref.append([word.deprel, word.text])
								tree_hyp = [[hyp_doc.words[listft[elid]].deprel, hyp_doc.words[hyp_doc.words[listft[elid]].head - 1].text if hyp_doc.words[listft[elid]].head > 0 else "root"]]
								for word in hyp_doc.words:
									if word.head == listft[elid] + 1:
										tree_hyp.append([word.deprel, word.text])
								
								denominator += len(tree_ref)
								for node_ref in tree_ref:
									edge = node_ref[0]
									node = node_ref[1]
									for node_hyp in tree_hyp:
										if edge == node_hyp[0] and node == node_hyp[1]:
											numerator+= 1
											break
							acc += (numerator / denominator) if denominator else 0
							break
	
	return acc / matched if matched > 0 else 0



def AWESOME(sent_src, sent_tgt):
	token_src, token_tgt = [[word] if word == "[UNK]" else tokenizer.tokenize(word) for word in sent_src] \
		, [[word] if word == "[UNK]" else tokenizer.tokenize(word) for word in sent_tgt]

	wid_src, wid_tgt = [tokenizer.convert_tokens_to_ids(x) for x in token_src] \
		, [tokenizer.convert_tokens_to_ids(x) for x in token_tgt]

	ids_src, ids_tgt \
		= tokenizer.prepare_for_model(list(itertools.chain(*wid_src)) \
		, return_tensors='pt', model_max_length=tokenizer.model_max_length, truncation=True)['input_ids'] \
		, tokenizer.prepare_for_model(list(itertools.chain(*wid_tgt)) \
		, return_tensors='pt', truncation=True, model_max_length=tokenizer.model_max_length)['input_ids'] \

	sub2word_map_src = []
	for s, word_list in enumerate(token_src):
		sub2word_map_src += [s for x in word_list]
	sub2word_map_tgt = []
	for t, word_list in enumerate(token_tgt):
		sub2word_map_tgt += [t for x in word_list]

	align_layer = 8
	threshold = 1e-3
	model.eval()

	with torch.no_grad():
		out_src = model(ids_src.unsqueeze(0), output_hidden_states=True)[2][align_layer][0, 1:-1]
		out_tgt = model(ids_tgt.unsqueeze(0), output_hidden_states=True)[2][align_layer][0, 1:-1]

		dot_prod = torch.matmul(out_src, out_tgt.transpose(-1, -2))

		softmax_srctgt = torch.nn.Softmax(dim=-1)(dot_prod)
		softmax_tgtsrc = torch.nn.Softmax(dim=-2)(dot_prod)

		softmax_inter = (softmax_srctgt > threshold)*(softmax_tgtsrc > threshold)

	align_subwords = torch.nonzero(softmax_inter, as_tuple=False)
	align_words = {}
	for s, t in align_subwords:
		if sub2word_map_src[s] not in align_words:
			align_words[sub2word_map_src[s]] = [sub2word_map_tgt[t]]
		else:
			align_words[sub2word_map_src[s]].append(sub2word_map_tgt[t])

	return align_words



def exact_match(l2, references, outputs, ids):
	correct = 0
	wrong = 0
	correctl = 0
	wrongl = 0
	for i, id in enumerate(ids):
		if id in references:
			c, w, cl, wl = compare_EXACT(outputs[i], references[id])
			correct += c
			wrong += w
			correctl += cl
			wrongl += wl

	print(f"Exact-Match Statistics")
	print(f"\tTotal correct: {correct}")
	print(f"\tTotal wrong: {wrong}")
	print(f"\tTotal correct (lemma): {correctl}")
	print(f"\tTotal wrong (lemma): {wrongl}")
	print(f"Exact-Match Accuracy: {(correct + correctl) / (correct + correctl + wrong + wrongl)}")

def exact_match_ENG(l2, references, outputs, ids):
	correct = 0
	wrong = 0
	correctl = 0
	wrongl = 0
	for i, id in enumerate(ids):
		if id in references:
			c, w, cl, wl = compare_EXACT_ENG(outputs[i], references[id])
			correct += c
			wrong += w
			correctl += cl
			wrongl += wl

	print(f"Exact-Match Statistics")
	print(f"\tTotal correct: {correct}")
	print(f"\tTotal wrong: {wrong}")
	print(f"\tTotal correct (lemma): {correctl}")
	print(f"\tTotal wrong (lemma): {wrongl}")
	print(f"Exact-Match Accuracy : {(correct + correctl) / (correct + correctl + wrong + wrongl)}")

def bleu(l2, references, outputs):
	references = [references]
	bleu = sacrebleu.corpus_bleu(outputs, references)
	print(f"BLEU score: {bleu.score}")

def ter_w_shift(l2, references, outputs, IDS_to_exclude=[]):
	ter = 0
	cnt = 0
	for i in range(len(outputs)):
		if i in IDS_to_exclude:
			continue
		ter += TER.ter(outputs[i].split(), references[i].split())
		cnt += 1
	print(f"1 - TER Score: {1 - (ter / cnt)}")

def mod_ter_w_shift(l2, references, outputs, nonreferences, ids, lc, IDS_to_exclude=[]):
	ter = 0.0
	for i, sid in enumerate(ids):
		if i in IDS_to_exclude:
			continue
		if sid in references:
			ter += compare_TER_w(outputs[i], references[sid], lc)
		else:
			ter += TER_modified.ter(outputs[i].split(), nonreferences[i].split(), lc)
		
	print(f"1 - TERm Score: {1 - (ter / len(ids))}")

def mod_ter_w_shift_ENG(l2, references, outputs, nonreferences, ids, lc, IDS_to_exclude=[]):
	ter = 0.0
	for i, sid in enumerate(ids):
		if i in IDS_to_exclude:
			continue
		if sid in references:
			ter += compare_TER_w_ENG(outputs[i], references[sid], lc)
		else:
			ter += TER_modified.ter(outputs[i].split(), nonreferences[i].split(), lc)
		
	print(f"1 - TERm Score: {1 - (ter / len(ids))}")

def exact_alignment_match_bleu(l2, references, outputs, ids, window):
	bleu_score = 0.0
	for i, id in enumerate(ids):
		if id in references:
			bleu_score += compare_exact_align_bleu(outputs[i], references[id], window)
	
	print(f"Exact Alignment Match BLEU score: {bleu_score / (len(references))}")

def exact_alignment_match_bleu_ENG(l2, references, outputs, ids, window):
	bleu_score = 0.0
	for i, id in enumerate(ids):
		if id in references:
			bleu_score += compare_exact_align_bleu_ENG(outputs[i], references[id], window)
	
	print(f"Exact Alignment Match BLEU score: {bleu_score / (len(references))}")

def exact_alignment_match(l2, references, outputs, ids, window):
	acc = 0.0
	high = open(f"result/Examples_high_en-{l2}.txt", "w")
	low = open(f"result/Examples_low_en-{l2}.txt", "w")
	for i, id in enumerate(ids):
		if id in references:
			acc1 = compare_exact_align(outputs[i], references[id], window)
			acc += acc1
	
	print(f"Exact Alignment Match: {acc / (len(references))}")

def exact_alignment_match_ENG(l2, references, outputs, ids, window):
	acc = 0.0
	high = open(f"result/Examples_high_{l2}-en.txt", "w")
	low = open(f"result/Examples_low_{l2}-en.txt", "w")
	for i, id in enumerate(ids):
		if id in references:
			acc1 = compare_exact_align_ENG(outputs[i], references[id], window)
			acc += acc1
	
	print(f"Exact Alignment Match: {acc / (len(references))}")

def exact_window_overlap_match(l2, references, outputs, ids, window):
	acc = 0.0
	for i, id in enumerate(ids):
		if id in references:
			if outputs[i] != "":
				acc1 = compare_exact_window_overlap(outputs[i], references[id], window)
			#if acc1 > 1:
			#	print("yo")
			acc += acc1
	
	print(f"\tExact Window Overlap Accuracy: {acc / (len(references))}")

def exact_window_overlap_match_ENG(l2, references, outputs, ids, window):
	acc = 0.0
	for i, id in enumerate(ids):
		if id in references:
			if outputs[i] != "":
				acc1 = compare_exact_window_overlap_ENG(outputs[i], references[id], window)
			if acc1 > 1:
				print("yo")
			acc += acc1
	
	print(f"\tExact Window Overlap Accuracy: {acc / (len(references))}")

def exact_alignment_match_UD(l2, references, outputs, ids):
	acc = 0.0
	for i, id in enumerate(ids):
		if id in references:
			acc += compare_exact_align_UD(outputs[i], references[id])
	
	print(f"Exact Match over UD Window Accuracy {acc / (len(references))}")

def exact_alignment_match_UD_ENG(l2, references, outputs, ids):
	acc = 0.0
	for i, id in enumerate(ids):
		if id in references:
			acc += compare_exact_align_UD_ENG(outputs[i], references[id])
	
	print(f"Exact Match over UD Window Accuracy {acc / (len(references))}")


parser = argparse.ArgumentParser()
parser.add_argument("--language", help="target language", type=str, default="")
parser.add_argument("--hypothesis", help="hypothesis file", type=str, default="")
parser.add_argument("--unmodified_ref_directory", help="directory where output will be saved", type=str, default="")
parser.add_argument("--id_directory", help="directory where id is located", type=str, default="")
parser.add_argument("--modified_ref_directory", help="directory where ref is located", type=str, default="")
parser.add_argument("--BLEU", help="", type=str, default="True")
parser.add_argument("--EXACT_MATCH", help="", type=str, default="True")
parser.add_argument("--MOD_TER", help="", type=str, default="True")
parser.add_argument("--WINDOW_OVERLAP", help="", type=str, default="True")
parser.add_argument("--TER", help="", type=str, default="False")
parser.add_argument("--ALIGN_EXACT", help="", type=str, default="False")
parser.add_argument("--ALIGN_BLEU", help="", type=str, default="False")
parser.add_argument("--ALIGN_UD", help="", type=str, default="False")
args = parser.parse_args()

l2 = args.language
windows = [2, 3]
model = transformers.BertModel.from_pretrained('bert-base-multilingual-cased')
tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-multilingual-cased')


SUPPORTED = False
try:
	stanza.download(l2, processors='tokenize,pos,lemma,depparse')
	l2_stanza = stanza.Pipeline(processors='tokenize,pos,lemma,depparse', lang=l2, use_gpu=True, tokenize_pretokenized=True)
	SUPPORTED = True
except:
	print(f"Language {l2} does not seem to be supported by Stanza -- or something went wrong with downloading the models.")
	print(f"Will continue without searching over lemmatized versions of the data.")
	SUPPORTED = False


ids = read_ids(args.id_directory)
outputs = read_outputs(args.hypothesis)
sentreferences = read_outputs(args.unmodified_ref_directory)
if l2 != "en":
	exactreferences = read_reference_data(args.modified_ref_directory)
	if args.BLEU == "True":
		bleu(l2, sentreferences, outputs)
	if args.EXACT_MATCH == "True":
		exact_match(l2, exactreferences, outputs, ids)
	if args.WINDOW_OVERLAP == "True":
		print("Window Overlap Accuracy :")
		for window in windows:
			print(f"\tWindow {window}:")
			exact_window_overlap_match(l2, exactreferences, outputs, ids, window)
	if args.TER == "True":
		ter_w_shift(l2, sentreferences, outputs)
	if args.MOD_TER == "True":
		mod_ter_w_shift(l2, exactreferences, outputs, sentreferences, ids, 2)
	if args.ALIGN_EXACT == "True":
		print("Accuracy Alignment Exact Order:")
		for window in windows:
			print(f"Window {window}:")
			exact_alignment_match(l2, exactreferences, outputs, ids, window)
	if args.ALIGN_BLEU == "True":
		print("Accuracy Alignment BLEU:")
		for window in windows:
			print(f"Window {window}:")
			exact_alignment_match_bleu(l2, exactreferences, outputs, ids, window)
	if args.ALIGN_UD == "True":
		print("Accuracy Alignment UD:")
		exact_alignment_match_UD(l2, exactreferences, outputs, ids)
else:
	exactreferences = read_reference_data_ENG(args.modified_ref_directory)
	if args.BLEU == "True":
		bleu(l2, sentreferences, outputs)
	if args.EXACT_MATCH == "True":
		exact_match_ENG(l2, exactreferences, outputs, ids)
	if args.WINDOW_OVERLAP == "True":
		print("Window Overlap Accuracy:")
		for window in windows:
			print(f"\tWindow {window}:")
			exact_window_overlap_match_ENG(l2, exactreferences, outputs, ids, window)
	if args.TER == "True":
		ter_w_shift(l2, sentreferences, outputs, [])
	if args.MOD_TER == "True":
		mod_ter_w_shift_ENG(l2, exactreferences, outputs, sentreferences, ids, 2)
	if args.ALIGN_EXACT == "True":
		print("Accuracy Alignment Exact Order:")
		for window in windows:
			print(f"Window {window}:")
			exact_alignment_match_ENG(l2, exactreferences, outputs, ids, window)
	if args.ALIGN_BLEU == "True":
		print("Accuracy Alignment BLEU:")
		for window in windows:
			print(f"Window {window}:")
			exact_alignment_match_bleu_ENG(l2, exactreferences, outputs, ids, window)
	if args.ALIGN_UD == "True":
		print("Accuracy Alignment UD:")
		exact_alignment_match_UD_ENG(l2, exactreferences, outputs, ids)
