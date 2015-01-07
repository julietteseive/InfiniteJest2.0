__author__ = 'JulietteSeive'

from gensim import corpora, models, similarities
from gensim.models import ldamodel
from itertools import izip
import nltk
import re
import os
import pdb
from textclean.textclean import textclean


start_tag = "<SEGMENT>"
end_tag = "</SEGMENT>"

PATH_TO_NEW_JEST = "jest-with-tags.txt"


def get_jest():
    return open(PATH_TO_NEW_JEST).readlines()


def is_close_tag(tag):
    return tag.startswith("</")


def strip_all_tags(seg_text):
    return strip_tags(seg_text, get_tags_in_text(seg_text))


def split_by_tags():
    j = open("jest-with-tags2.txt", "r")
    jest = j.read()
    jest = re.sub(r'\n\s*\n', " ", jest)
    jest = jest.decode('utf-8')
    # print jest
    segments = []
    create_segments(jest, segments)
    print (segments)
    return segments


def create_segments(jest, segments):
    # start_index = jest.index(start_tag)
    end_index = jest.index(end_tag)
    first_segment = jest[9:end_index]
    segments.append(first_segment)
    if len(jest) > (end_index + 10):
        jest = jest[end_index + 10:]
        create_segments(jest, segments)




def get_NERs(segments):
    #NER_dict = {}  # map entities to counts (i.e., # of occurences in this seg)
    #NERs_to_types = {}  # map the NERs to the kinds of things they are
    NER_dicts = []
    NERs_types = []

    for text, segment_text in enumerate(segments):
        NER_dict = {}
        NERs_to_types = {}
        print text
        print segments[text]
        # tokenize, then POS text
        pos_tagged_seg = nltk.pos_tag(nltk.word_tokenize(segments[text]))
        print pos_tagged_seg

        # and now the NER
        NERd_seg = nltk.ne_chunk(pos_tagged_seg)

        # kind of hacky, but this is how I'm parsing
        # the induced tree structure
        for subtree in NERd_seg:
            # then this is an NER
            if type(subtree) == nltk.tree.Tree:
                # ignoring the *type* of NER for now -- i can't think of a
                # case in which we'd care (typically, entities with the same
                # name *ought* to be of the same type, I think...)
                entity = subtree[0][0]  # this parses out the token (entity) itself
                entity_type = subtree.label()
                # if we've already encountered it, just bump the count
                if entity in NER_dict:
                    NER_dict[entity] += 1
                else:
                    NER_dict[entity] = 1
                    NERs_to_types[entity] = subtree.label()  # ## going to assume we always get this correct, I guess

        NER_dicts.append(NER_dict)
        NERs_types.append(NERs_to_types)
    return NER_dicts, NERs_types


def populate_list(n):
    word_list = []
    for segment in n:
        segment_wordlist = []
        for word, count in segment.items():
            segment_wordlist.extend([word]*count)
        word_list.append(segment_wordlist)
    return word_list



def extract_topics(text, numTopics=5):  # list of entities, arbitrary number of topics


    dict1 = corpora.Dictionary(text)  # generate dictionary
    # dict1.compactify()
    corpus = [dict1.doc2bow(t) for t in text]

    #printing documents and most probable topics for each doc
    lda = ldamodel.LdaModel(corpus, id2word=dict1, num_topics=5)
    corpus_lda = lda[corpus]

    for i in lda.show_topic(topicid=2, topn=5):
        print i

    x = lda.print_topics(3)

    print x

def main():
    x = split_by_tags()
    n = get_NERs(x)[0]
    print(n)
    l = populate_list(n)
    print l

    print 'Printing text...'
    #print(text)
    lda_output = extract_topics(l, numTopics=5)



main()