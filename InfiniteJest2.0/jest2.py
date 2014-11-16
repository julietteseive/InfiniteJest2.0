__author__ = 'JulietteSeive'

import nltk
import re


def get_tags_in_text(text):
    p = re.compile(r'<.*?>') # finds the *closing* tags
    all_tags = p.findall(text)
    tags = [x for x in list(set(all_tags))]
    # which are close tags?
    return all_tags



def get_NERs(path_to_seg):
    NER_dict = {} # map entities to counts (i.e., # of occurences in this seg)
    NERs_to_types = {} # map the NERs to the kinds of things they are

    seg_text = open(path_to_seg).read()

    # strip *all* tags
    seg_text = strip_tags(seg_text, get_tags_in_text(seg_text))

    # tokenize, then POS text
    pos_tagged_seg = nltk.pos_tag(nltk.word_tokenize(seg_text))

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
            entity = subtree[0][0] # this parses out the token (entity) itself
            entity_type = subtree.node
            # if we've already encountered it, just bump the count
            if entity in NER_dict:
                NER_dict[entity] += 1
            else:
                NER_dict[entity] = 1
                NERs_to_types[entity] = subtree.node ### going to assume we always get this correct, I guess

    return NER_dict, NERs_to_types