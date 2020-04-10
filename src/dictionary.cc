/**
 * Copyright (c) 2018-present, Yerai Doval Mosquera.
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "dictionary.h"

#include <assert.h>
#include <pthread.h>

#include <utf8proc.h>
#include <functional>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <iterator>
#include <cmath>
#include <stdexcept>
#include <random>
#include "VpTree.hpp"

#define NUM_THREADS 48 

namespace fasttext {

const std::string Dictionary::EOS = "</s>";
const std::string Dictionary::BOW = "<";
const std::string Dictionary::EOW = ">";
std::default_random_engine generator; 

Dictionary::Dictionary(std::shared_ptr<Args> args) : args_(args),
  word2int_(MAX_VOCAB_SIZE, -1), size_(0), nwords_(0), nlabels_(0),
  ntokens_(0), pruneidx_size_(-1) {}

Dictionary::Dictionary(std::shared_ptr<Args> args, std::istream& in) : args_(args),
  size_(0), nwords_(0), nlabels_(0), ntokens_(0), pruneidx_size_(-1) {
  load(in);
}

// Returns true if p_char is a vowel
bool is_vowel(const char p_char)
{

    constexpr char vowels[] = { 'a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U' };
    return std::find(std::begin(vowels), std::end(vowels), p_char) != std::end(vowels);
}

std::string remove_vowel(std::string st) 
{
    // Moves all the characters for which `is_vowel` is true to the back
    //  and returns an iterator to the first such character
    auto to_erase = std::remove_if(st.begin(), st.end(), is_vowel);

    // Actually remove the unwanted characters from the string
    st.erase(to_erase, st.end());
    return st;
}


int32_t Dictionary::find(const std::string& w) const {
  return find(w, hash(w));
}

int32_t Dictionary::find(const std::string& w, uint32_t h) const {
  int32_t word2intsize = word2int_.size();
  int32_t id = h % word2intsize;
  while (word2int_[id] != -1 && words_[word2int_[id]].word != w) {
    id = (id + 1) % word2intsize;
  }
  return id;
}

void Dictionary::add(const std::string& w) {
  int32_t h = find(w);
  ntokens_++;
  if (word2int_[h] == -1) {
    entry e;
    e.id = word2int_[h];
    e.word = w;
    e.count = 1;
    e.type = getType(w);
    words_.push_back(e);
    word2int_[h] = size_++;
  } else {
    words_[word2int_[h]].count++;
  }
}

int32_t Dictionary::nwords() const {
  return nwords_;
}

int32_t Dictionary::nlabels() const {
  return nlabels_;
}

int64_t Dictionary::ntokens() const {
  return ntokens_;
}

const std::vector<int32_t>& Dictionary::getSubwords(int32_t i) const {
  assert(i >= 0);
  assert(i < nwords_);
  return words_[i].subwords;
}

const std::vector<int32_t> Dictionary::getSubwords(
    const std::string& word) const {
  int32_t i = getId(word);
  if (i >= 0) {
    return getSubwords(i);
  }
  std::vector<int32_t> ngrams;
  if (word != EOS) {
    computeSubwords(BOW + word + EOW, ngrams);
  }
  return ngrams;
}

const std::vector<int32_t> Dictionary::getSubwordsNoSkip(
    const std::string& word) const {
  int32_t i = getId(word);
  if (i >= 0) {
    return getSubwords(i);
  }
  std::vector<int32_t> ngrams;
  if (word != EOS) {
    computeSubwordsNoSkip(BOW + word + EOW, ngrams);
  }
  return ngrams;
}

void Dictionary::getSubwords(const std::string& word,
                           std::vector<int32_t>& ngrams,
                           std::vector<std::string>& substrings) const {
  int32_t i = getId(word);
  ngrams.clear();
  substrings.clear();
  if (i >= 0) {
    ngrams.push_back(i);
    substrings.push_back(words_[i].word);
  }
  if (word != EOS) {
    computeSubwords(BOW + word + EOW, ngrams, substrings);
  }
}

void Dictionary::getSubwordsNoSkip(const std::string& word,
                           std::vector<int32_t>& ngrams,
                           std::vector<std::string>& substrings) const {
  int32_t i = getId(word);
  ngrams.clear();
  substrings.clear();
  if (i >= 0) {
    ngrams.push_back(i);
    substrings.push_back(words_[i].word);
  }
  if (word != EOS) {
    computeSubwordsNoSkip(BOW + word + EOW, ngrams, substrings);
  }
}

std::vector<int32_t> Dictionary::getReps(int32_t id, int palt) {    return words_[id].similar.reps;
}

int32_t Dictionary::getRepSym() {    return getId("<<SPECIAL_REP_SYMBOL>>");
}


std::vector<int32_t> Dictionary::getPhonSimilar(int32_t id, int palt) {    return words_[id].similar.phonetic;
}

std::vector<int32_t> Dictionary::getSimilar(int32_t id, int palt) {  

  return words_[id].similar.ortographic;
}

bool Dictionary::discard(int32_t id, real rand) const {
  assert(id >= 0);
  assert(id < nwords_);
  if (args_->model == model_name::sup) return false;
  return rand > pdiscard_[id];
}

int32_t Dictionary::getId(const std::string& w, uint32_t h) const {
  int32_t id = find(w, h);
  return word2int_[id];
}

int32_t Dictionary::getId(const std::string& w) const {
  int32_t h = find(w);
  return word2int_[h];
}

entry_type Dictionary::getType(int32_t id) const {
  assert(id >= 0);
  assert(id < size_);
  return words_[id].type;
}

entry_type Dictionary::getType(const std::string& w) const {
  return (w.find(args_->label) == 0) ? entry_type::label : entry_type::word;
}

sim Dictionary::getWordSimilar(int32_t id) const {
  assert(id >= 0);
  assert(id < size_);
  return words_[id].similar;
}

std::string Dictionary::getWord(int32_t id) const {
  assert(id >= 0);
  assert(id < size_);
  return words_[id].word;
}

int64_t Dictionary::getCount(int32_t id) const {
  assert(id >= 0);
  assert(id < size_);
  return words_[id].count;
}

uint32_t Dictionary::hash(const std::string& str) const {
  uint32_t h = 2166136261;
  for (size_t i = 0; i < str.size(); i++) {
    h = h ^ uint32_t(str[i]);
    h = h * 16777619;
  }
  return h;
}

void Dictionary::computeSubwords(const std::string& oword,
                               std::vector<int32_t>& ngrams,
                               std::vector<std::string>& substrings) const {
  for (size_t i = 0; i < oword.size(); i++) {
    std::string ngram;
    if ((oword[i] & 0xC0) == 0x80) continue;
    for (size_t j = i, n = 1; j < oword.size() && n <= args_->maxn; n++) {
      ngram.push_back(oword[j++]);
      while (j < oword.size() && (oword[j] & 0xC0) == 0x80) {
        ngram.push_back(oword[j++]);
      }
      if (n >= args_->minn && !(n == 1 && (i == 0 || j == oword.size()))) {
        int32_t h = hash(ngram) % args_->bucket;
        ngrams.push_back(nwords_ + h);
        substrings.push_back(ngram);
      }
    }
  }
}

void Dictionary::computeSubwordsNoSkip(const std::string& oword,
                               std::vector<int32_t>& ngrams,
                               std::vector<std::string>& substrings) const {
  for (size_t i = 0; i < oword.size(); i++) {
    std::string ngram;
    if ((oword[i] & 0xC0) == 0x80) continue;
    for (size_t j = i, n = 1; j < oword.size() && n <= args_->maxn; n++) {
      ngram.push_back(oword[j++]);
      while (j < oword.size() && (oword[j] & 0xC0) == 0x80) {
        ngram.push_back(oword[j++]);
      }
      if (n >= args_->minn && !(n == 1 && (i == 0 || j == oword.size()))) {
        int32_t h = hash(ngram) % args_->bucket;
        ngrams.push_back(nwords_ + h);
        substrings.push_back(ngram);
      }
    }
  }
}

void Dictionary::computeSubwords(const std::string& oword,
                               std::vector<int32_t>& ngrams) const {
  std::vector<std::string> substrings;
  for (size_t i = 0; i < oword.size(); i++) {
    std::string ngram;
    if ((oword[i] & 0xC0) == 0x80) continue;
    for (size_t j = i, n = 1; j < oword.size() && n <= args_->maxn; n++) {
      ngram.push_back(oword[j++]);
      while (j < oword.size() && (oword[j] & 0xC0) == 0x80) {
        ngram.push_back(oword[j++]);
      }
      if (n >= args_->minn && !(n == 1 && (i == 0 || j == oword.size()))) {
        int32_t h = hash(ngram) % args_->bucket;
        pushHash(ngrams, h);
      }
    }
  }
}

void Dictionary::computeSubwordsNoSkip(const std::string& oword,
                               std::vector<int32_t>& ngrams) const {
  for (size_t i = 0; i < oword.size(); i++) {
    std::string ngram;
    if ((oword[i] & 0xC0) == 0x80) continue;
    for (size_t j = i, n = 1; j < oword.size() && n <= args_->maxn; n++) {
      ngram.push_back(oword[j++]);
      while (j < oword.size() && (oword[j] & 0xC0) == 0x80) {
        ngram.push_back(oword[j++]);
      }
      if (n >= args_->minn && !(n == 1 && (i == 0 || j == oword.size()))) {
        int32_t h = hash(ngram) % args_->bucket;
        pushHash(ngrams, h);
      }
    }
  }
}

void Dictionary::initNgrams() {
  for (size_t i = 0; i < size_; i++) {
    std::string word = BOW + words_[i].word + EOW;
    words_[i].subwords.clear();
    words_[i].subwords.push_back(i);
      computeSubwords(word, words_[i].subwords);
  }
}

bool Dictionary::readWord(std::istream& in, std::string& word) const
{
  char c;
  std::streambuf& sb = *in.rdbuf();
  word.clear();
  while ((c = sb.sbumpc()) != EOF) {
    if (c == ' ' || c == '\n' || c == '\r' || c == '\t' || c == '\v' ||
        c == '\f' || c == '\0') {
      if (word.empty()) {
        if (c == '\n') {
          word += EOS;
          return true;
        }
        continue;
      } else {
        if (c == '\n')
          sb.sungetc();
        return true;
      }
    }
    word.push_back(c);
  }
  // trigger eofbit
  in.get();
  return !word.empty();
}

void Dictionary::readExtraDict(std::istream& in) {
  //TODO
}

double lev_dist(const entry& s1e, const entry& s2e) 
{
  std::string s1 = s1e.word;
  std::string s2 = s2e.word;
        const std::size_t len1 = s1.size(), len2 = s2.size();
	std::vector<unsigned int> col(len2+1), prevCol(len2+1);
	
	for (unsigned int i = 0; i < prevCol.size(); i++)
		prevCol[i] = i;
	for (unsigned int i = 0; i < len1; i++) {
		col[0] = i+1;
		for (unsigned int j = 0; j < len2; j++)
                        // note that std::min({arg1, arg2, arg3}) works only in C++11,
                        // for C++98 use std::min(std::min(arg1, arg2), arg3)
			col[j+1] = std::min({ prevCol[1 + j] + 1, col[j] + 1, prevCol[j] + (s1[i]==s2[j] ? 0 : 1) });
		col.swap(prevCol);
	}
	return prevCol[len2];
}

double dam_lev_dist(const entry& s1e, const entry& s2e) {
  std::string s1 = s1e.word;
  std::string s2 = s2e.word;
  size_t size1 = s1.size();
  size_t size2 = s2.size();
  size_t d[size1 + 1][size2 + 1];
  for (int i = 0; i <= size1; i ++)
    d[i][0] = i;
  for (int i = 0; i <= size2; i ++)
    d[0][i] = i;

  int cost = 0; 
  for (int i = 1; i <= size1; i ++)
    for (int j = 1; j <= size2; j ++)
    {      
      cost = (s1[i - 1] == s2[j - 1]) ? 0 : 1 ;
      if ( (i > 1) && (j > 1) && (s1[i] == s2[j - 1]) && (s1[i - 1] == s2[j]))
      {
        size_t a = std::min(d[i - 1][j], d[i][j - 1] + 1);
        size_t b = std::min(d[i][j] + cost, d[i - 2][j - 2]);
        d[i][j] = std::min(a, b);
      }
      else
      {
        d[i][j] = std::min(std::min(d[i][j -1] + 1, d[i - 1][j] + 1), d[i - 1][j - 1] + cost);
      }
    }
  return (double)(d[size1][size2]);
}

void lowercase(std::string str) {

    unsigned long i = 0;
    unsigned long j = 0;
    unsigned char c;

    while ((c = str[i++]) != '\0')
    {
            str[j++] = tolower(c);
    }
    str[j] = '\0';
}

void stringRemoveNonAlphaNumLow(unsigned char *str) {
    unsigned long i = 0;
    unsigned long j = 0;
    unsigned char c;

    while ((c = str[i++]) != '\0')
    {
        if (isalnum(c))
        {
            str[j++] = tolower(c);
        }
    }
    str[j] = '\0';
//    return str;
}

void removeReps(unsigned char *str) {
	int i, j, len, len1;

    for (len=0; str[len]!='\0'; len++);
 
    len1=0;
 
    for (i=0; i < (len-len1);)
    {
        if(str[i] == str[i+1])
        {
            for(j = i; j < (len-len1); j++)
                str[j] = str[j+1];
            len1++;
        }
        else
        {
            i++;
        }
    }
}

std::string proc(const std::string word) {

  utf8proc_uint8_t* pUTF = utf8proc_NFKD( (const unsigned char*)word.c_str() );
  stringRemoveNonAlphaNumLow(pUTF);
	removeReps(pUTF);
  std::string strUTF;
  if (pUTF) {
    strUTF = (char*)pUTF;
    free(pUTF);
  }
	return strUTF;
}

std::vector<std::string> getSkips(const std::string ooword) {
	std::vector<std::string> acc;
	std::string oword = proc(ooword);
  	for (size_t y = 0; y < oword.size(); y++) {
  		std::string skipgram;
  		for (size_t yy = 0; yy < oword.size(); yy++) {
  			if (yy == y) {
				continue;
			} else {
				skipgram.push_back(oword[yy]);
			}
  	} 
		if (skipgram.size() > 0) {
			acc.push_back(skipgram);
		}
  }
	return acc;
}

std::vector<std::string> getNSkips(std::string oword, int n) {
	std::vector<std::string> acc;
	if (n == 0) return acc;
	std::vector<std::string> skips = getSkips(oword);
	acc.insert( acc.end(), skips.begin(), skips.end() );
	for (int i = 0; i < skips.size(); i++) {
		std::vector<std::string> skipskips = getNSkips(skips[i], n - 1);
		acc.insert( acc.end(), skipskips.begin(), skipskips.end() );
	}
	return acc;
}

std::vector<std::string> getAllSkips(std::string oword) {
	std::vector<std::string> acc;
	if (oword.size() == 0) return acc;
	std::vector<std::string> skips = getSkips(oword);
	acc.insert( acc.end(), skips.begin(), skips.end() );
	for (int i = 0; i < skips.size(); i++) {
		std::vector<std::string> skipskips = getAllSkips(skips[i]);
		acc.insert( acc.end(), skipskips.begin(), skipskips.end() );
	}
	return acc;
}

struct thread_data {
  int tid;
  int32_t* nwords;
  std::vector<entry>* words;
  std::string* output_file;
  std::function<double (const entry&, const entry&)> dist_metric;
  std::function<int32_t (const std::string&)> get_id;
  Dictionary* dict;
  VpTree<entry, dam_lev_dist>* tree;
};

void *splitLoop(void *args) {
   const double MAX_DIST = 2;
   struct thread_data* data;
   data = (struct thread_data *) args;
   int32_t nw = *(data->nwords);
   std::string output_file = *(data->output_file);
   int perjob = std::ceil(*(data->nwords) / NUM_THREADS);
   int32_t local_tid = data->tid;
   int start = local_tid * perjob;
   int end = std::min(start + perjob, *(data->nwords));
   double dist;
   std::vector<entry> ws = *(data->words);
   std::vector<entry>* wsptr = data->words;
   for(int32_t i = start; i < end; i++) {
     entry curr_word = (*wsptr)[i];
     std::vector<entry> similar_words;
     std::vector<double> distances;
     data->tree->search((*wsptr)[i], MAX_DIST, &similar_words, &distances);
     for (int j = 0; j < similar_words.size(); j++) {
	if (distances[j] > 0) {
		(*wsptr)[i].similar.ortographic.push_back(data->dict->getId(similar_words[j].word));
        }
     }
   }
}

void Dictionary::initOrtoSim() {
   for(int32_t i = 0; i < size_; i++) {
     entry* curr_word = &words_[i];
     std::vector<std::string> similar_words = getSkips((*curr_word).word);

     for (int j = 0; j < similar_words.size(); j++) {
        if (getId(similar_words[j]) == -1) { 
        	continue;
	}
        (*curr_word).similar.ortographic.push_back(getId(similar_words[j]));
     }
   }
}

void Dictionary::readFromFile(std::istream& in) {
  std::string oword;
  int64_t minThreshold = 1;
  while (readWord(in, oword)) {
	    add(oword);
	      std::cerr << "\33[2K\rRead " << oword << " " << ntokens_  / 1000000 << "M words" << std::flush;
	    if (size_ > 0.75 * MAX_VOCAB_SIZE) {
	      minThreshold++;
	      threshold(minThreshold, minThreshold);
	    }
  }

  int64_t static_size = size_; 
  std::vector<std::string> pivots_ortho;
  for (int y = 0; y < static_size; y++) {
    oword = words_[y].word;
    std::vector<std::string> words = getSkips(oword); 
  
    pivots_ortho.insert(pivots_ortho.end(), words.begin(), words.end());    
  }
  std::cerr << std::endl << "Pivots obtained: " << pivots_ortho.size() << std::endl;

    minThreshold = 1;
    for (int i = 0; i < pivots_ortho.size(); i++) {
	      			std::cerr << "\rPivots: " << (i/(float)pivots_ortho.size())*100 << std::flush;
            			add(pivots_ortho[i]);
	    if (size_ > 0.75 * MAX_VOCAB_SIZE) {
	      minThreshold++;
	      threshold(minThreshold, minThreshold);
	    }
  
    }

  std::cerr << "Words" << std::endl;
  threshold(args_->minCount, args_->minCountLabel);
  std::cerr << "Threshold" << std::endl;
  initTableDiscard();
  std::cerr << "Table" << std::endl;
  initNgrams();
  std::cerr << "Ngrams" << std::endl;
  if (args_->verbose > 0) {
    std::cerr << "\rRead " << ntokens_  / 1000000 << "M words" << std::endl;
    std::cerr << "Number of words:  " << nwords_ << std::endl;
    std::cerr << "Number of labels: " << nlabels_ << std::endl;
  }
  if (size_ == 0) {
    throw std::invalid_argument(
        "Empty vocabulary. Try a smaller -minCount value.");
  }
  initOrtoSim();
}

void Dictionary::threshold(int64_t t, int64_t tl) {
  sort(words_.begin(), words_.end(), [](const entry& e1, const entry& e2) {
      if (e1.type != e2.type) return e1.type < e2.type;
      return e1.count > e2.count;
    });
  words_.erase(remove_if(words_.begin(), words_.end(), [&](const entry& e) {
        return (e.type == entry_type::word && e.count < t) ||
               (e.type == entry_type::label && e.count < tl);
      }), words_.end());
  words_.shrink_to_fit();
  size_ = 0;
  nwords_ = 0;
  nlabels_ = 0;
  std::fill(word2int_.begin(), word2int_.end(), -1);
  for (auto it = words_.begin(); it != words_.end(); ++it) {
    int32_t h = find(it->word);
    word2int_[h] = size_++;
    if (it->type == entry_type::word) nwords_++;
    if (it->type == entry_type::label) nlabels_++;
  }
}

void Dictionary::initTableDiscard() {
  pdiscard_.resize(size_);
  for (size_t i = 0; i < size_; i++) {
    real f = real(words_[i].count) / real(ntokens_);
    pdiscard_[i] = std::sqrt(args_->t / f) + args_->t / f;
  }
}

std::vector<int64_t> Dictionary::getCounts(entry_type type) const {
  std::vector<int64_t> counts;
  for (auto& w : words_) {
    if (w.type == type) counts.push_back(w.count);
  }
  return counts;
}

void Dictionary::addWordNgrams(std::vector<int32_t>& line,
                               const std::vector<int32_t>& hashes,
                               int32_t n) const {
  for (int32_t i = 0; i < hashes.size(); i++) {
    uint64_t h = hashes[i];
    for (int32_t j = i + 1; j < hashes.size() && j < i + n; j++) {
      h = h * 116049371 + hashes[j];
      pushHash(line, h % args_->bucket);
    }
  }
}

void Dictionary::addSubwords(std::vector<int32_t>& line,
                             const std::string& token,
                             int32_t wid) const {
  if (wid < 0) { // out of vocab
    if (token != EOS) {
      computeSubwords(BOW + token + EOW, line);
    }
  } else {
    if (args_->maxn <= 0) { // in vocab w/o subwords
      line.push_back(wid);
    } else { // in vocab w/ subwords
      const std::vector<int32_t>& ngrams = getSubwords(wid);
      line.insert(line.end(), ngrams.cbegin(), ngrams.cend());
    }
  }
}

void Dictionary::reset(std::istream& in) const {
  if (in.eof()) {
    in.clear();
    in.seekg(std::streampos(0));
  }
}

int32_t Dictionary::getLine(std::istream& in,
                            std::vector<int32_t>& words,
                            std::minstd_rand& rng) const {
  std::uniform_real_distribution<> uniform(0, 1);
  std::string token;
  int32_t ntokens = 0;

  reset(in);
  words.clear();
  while (readWord(in, token)) {
    int32_t h = find(token);
    int32_t wid = word2int_[h];
    if (wid < 0) continue;

    ntokens++;
    if (getType(wid) == entry_type::word && !discard(wid, uniform(rng))) {
      words.push_back(wid);
    }
    if (ntokens > MAX_LINE_SIZE || token == EOS) break;
  }
  return ntokens;
}

int32_t Dictionary::getLine(std::istream& in,
                            std::vector<int32_t>& words,
                            std::vector<int32_t>& labels) const {
  std::vector<int32_t> word_hashes;
  std::string token;
  int32_t ntokens = 0;

  reset(in);
  words.clear();
  labels.clear();
  while (readWord(in, token)) {
    uint32_t h = hash(token);
    int32_t wid = getId(token, h);
    entry_type type = wid < 0 ? getType(token) : getType(wid);

    ntokens++;
    if (type == entry_type::word) {
      addSubwords(words, token, wid);
      word_hashes.push_back(h);
    } else if (type == entry_type::label && wid >= 0) {
      labels.push_back(wid - nwords_);
    }
    if (token == EOS) break;
  }
  addWordNgrams(words, word_hashes, args_->wordNgrams);
  return ntokens;
}

void Dictionary::pushHash(std::vector<int32_t>& hashes, int32_t id) const {
  if (pruneidx_size_ == 0 || id < 0) return;
  if (pruneidx_size_ > 0) {
    if (pruneidx_.count(id)) {
      id = pruneidx_.at(id);
    } else {
      return;
    }
  }
  hashes.push_back(nwords_ + id);
}

std::string Dictionary::getLabel(int32_t lid) const {
  if (lid < 0 || lid >= nlabels_) {
    throw std::invalid_argument(
        "Label id is out of range [0, " + std::to_string(nlabels_) + "]");
  }
  return words_[lid + nwords_].word;
}

void Dictionary::save(std::ostream& out) const {
  out.write((char*) &size_, sizeof(int32_t));
  out.write((char*) &nwords_, sizeof(int32_t));
  out.write((char*) &nlabels_, sizeof(int32_t));
  out.write((char*) &ntokens_, sizeof(int64_t));
  out.write((char*) &pruneidx_size_, sizeof(int64_t));
  for (int32_t i = 0; i < size_; i++) {
    entry e = words_[i];
    out.write(e.word.data(), e.word.size() * sizeof(char));
    out.put(0);
    out.write((char*) &(e.count), sizeof(int64_t));
    out.write((char*) &(e.type), sizeof(entry_type));
  }
  for (const auto pair : pruneidx_) {
    out.write((char*) &(pair.first), sizeof(int32_t));
    out.write((char*) &(pair.second), sizeof(int32_t));
  }
}

void Dictionary::load(std::istream& in) {
  words_.clear();
  in.read((char*) &size_, sizeof(int32_t));
  in.read((char*) &nwords_, sizeof(int32_t));
  in.read((char*) &nlabels_, sizeof(int32_t));
  in.read((char*) &ntokens_, sizeof(int64_t));
  in.read((char*) &pruneidx_size_, sizeof(int64_t));
  for (int32_t i = 0; i < size_; i++) {
    char c;
    entry e;
    while ((c = in.get()) != 0) {
      e.word.push_back(c);
    }
    in.read((char*) &e.count, sizeof(int64_t));
    in.read((char*) &e.type, sizeof(entry_type));
    words_.push_back(e);
  }
  pruneidx_.clear();
  for (int32_t i = 0; i < pruneidx_size_; i++) {
    int32_t first;
    int32_t second;
    in.read((char*) &first, sizeof(int32_t));
    in.read((char*) &second, sizeof(int32_t));
    pruneidx_[first] = second;
  }
  initTableDiscard();
  initNgrams();

  int32_t word2intsize = std::ceil(size_ / 0.7);
  word2int_.assign(word2intsize, -1);
  for (int32_t i = 0; i < size_; i++) {
    word2int_[find(words_[i].word)] = i;
  }
}

void Dictionary::prune(std::vector<int32_t>& idx) {
  std::vector<int32_t> words, ngrams;
  for (auto it = idx.cbegin(); it != idx.cend(); ++it) {
    if (*it < nwords_) {words.push_back(*it);}
    else {ngrams.push_back(*it);}
  }
  std::sort(words.begin(), words.end());
  idx = words;

  if (ngrams.size() != 0) {
    int32_t j = 0;
    for (const auto ngram : ngrams) {
      pruneidx_[ngram - nwords_] = j;
      j++;
    }
    idx.insert(idx.end(), ngrams.begin(), ngrams.end());
  }
  pruneidx_size_ = pruneidx_.size();

  std::fill(word2int_.begin(), word2int_.end(), -1);

  int32_t j = 0;
  for (int32_t i = 0; i < words_.size(); i++) {
    if (getType(i) == entry_type::label || (j < words.size() && words[j] == i)) {
      words_[j] = words_[i];
      word2int_[find(words_[j].word)] = j;
      j++;
    }
  }
  nwords_ = words.size();
  size_ = nwords_ +  nlabels_;
  words_.erase(words_.begin() + size_, words_.end());
  initNgrams();
}

void Dictionary::dump(std::ostream& out) const {
  out << words_.size() << std::endl;
  for (auto it : words_) {
    std::string entryType = "word";
    if (it.type == entry_type::label) {
      entryType = "label";
    }
    out << it.word << " " << it.count << " " << entryType << std::endl;
  }
}

}
