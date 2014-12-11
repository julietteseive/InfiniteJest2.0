__author__ = 'JulietteSeive'

from gensim import corpora, models, similarities
from gensim.models import ldamodel
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
    #print jest
    segments = []
    create_segments(jest, segments)
    print (segments)
    return segments



def create_segments(jest, segments):
    #start_index = jest.index(start_tag)
    end_index = jest.index(end_tag)
    first_segment = jest[9:end_index]
    segments.append(first_segment)
    if len(jest) > (end_index + 10):
        jest = jest[end_index + 10:]
        create_segments(jest, segments)







"""


def get_tags_in_text(text):
    p = re.compile(r'<.*?>') # finds the *closing* tags
    all_tags = p.findall(text)
    tags = [x for x in list(set(all_tags))]
    # which are close tags?
    return all_tags


def strip_tags(seg, tags):
    stripped = seg
    for tag in tags:
        if type(seg) == type([]):
            # if a list of lines was passed in,
            # join them
            stripped = "\n".join(seg)
        stripped = stripped.replace(tag, "")
        # also the closing tag
        stripped = stripped.replace(tag.replace("<", "</"), "")

    return stripped

def get_tags_for_segments():
    '''
    return a tuple of segments and their tags; the tags will be removed
    from the text of the segments
    '''
    # the way to do this is a forward pass through the whole text
    # keeping track of open tags
    open_tags, tags_for_segments = [], []
    segments = get_segments()
    tags_opened_d = {}
    for i, segment in enumerate(segments):
        print "on segment %s. open tags: %s" % (i, open_tags)
        #pdb.set_trace()
        tags_in_segment = get_tags_in_text("\n ".join(segment))
        tags_closed_in_segment = [t for t in tags_in_segment if is_close_tag(t)]
        tags_opened_in_segment = [t for t in tags_in_segment if not is_close_tag(t)]

        # keep a record of when we opened each tag
        for new_tag in tags_opened_in_segment:
            # don't overwrite tags
            if not new_tag in tags_opened_d:
                tags_opened_d[new_tag] = i
            if new_tag in open_tags:
                pdb.set_trace()

        # take the already open tags, and anything opened in this segment
        tags_in_segment = open_tags + tags_opened_in_segment
        tags_for_segments.append(list(tags_in_segment))

        open_tags = tags_in_segment

        #
        # now remove tags from open_tags that were closed in this segment
        for tag in tags_closed_in_segment:
            try:
                cur_tag = tag.replace("</", "<")
                open_tags.remove(cur_tag)
                if cur_tag not in open_tags and cur_tag in tags_opened_d.keys():
                    # remove it from the dictionary
                    tags_opened_d.pop(cur_tag)
            except:
                print "whoops."
                pdb.set_trace()

    return (tags_for_segments, segments, tags_opened_d)



def get_segments(strip_tags=False):
    ''' returns all segments in text '''
    segments = []
    cur_segment = []
    JEST = get_jest()
    for line in JEST:
        if line == "\n":
            # blank line; this is a segment
            segments.append(cur_segment)
            cur_segment = []
        else:
            cur_segment.append(line)

    if strip_tags:
        # going to join the lines comprising segments here;
        # not sure if we should *always* do this in this method?
        # otherwise each segment is a list of lines
        segments = ["\n".join(segment) for segment in segments]
        segments = [strip_all_tags(segment) for segment in segments]

    return segments

def spit_out_text_for_segs(tags_we_care_about=["<Segment>"]):
    tags_for_segs, segs, opened_d = get_tags_for_segments()

    # strip the tags out of all the docs
    stripped_segs = []
    for tags_for_seg, seg in zip(tags_for_segs, segs):
        stripped_segs.append(strip_tags(seg, tags_for_seg))

    # make dirs for those tags we care about
    for tag in tags_we_care_about:
        try:
            os.mkdir(_remove_bracks(tag))
        except:
            print "ah! couldn't make directory for %s. already exists?" % tag

    # ok, now write out text
    i = 0
    for seg, tags_for_seg in zip(stripped_segs, tags_for_segs):
        for tag_p in tags_we_care_about:
            if tag_p in tags_for_seg:
                fout = open("%s/%s" % (_remove_bracks(tag_p), i), 'w')
                fout.write(seg)
                fout.close()
        i+=1
    print "done!"

"""

def get_NERs(segments):
    NER_dict = {} # map entities to counts (i.e., # of occurences in this seg)
    NERs_to_types = {} # map the NERs to the kinds of things they are

    for text in range(len(segments)-1):
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
                entity = subtree[0][0] # this parses out the token (entity) itself
                entity_type = subtree.label()
            # if we've already encountered it, just bump the count
                if entity in NER_dict:
                    NER_dict[entity] += 1
                else:
                    NER_dict[entity] = 1
                    NERs_to_types[entity] = subtree.label() ### going to assume we always get this correct, I guess

        return NER_dict #,NERs_to_types

def _remove_bracks(tag):
    return tag.replace("<", "").replace(">", "")


def extract_topics(text, numTopics = 5): # list of entities, arbitrary number of topics

    dict1 = corpora.Dictionary(text) # generate dictionary
    corpus = [dict1.doc2bow(t) for t in text]

    lda = models.ldamodel.LdaModel(corpus, num_topics=numTopics) # generate LDA model
    i = 0

    #print the topics

    for topic in lda.show_topic(topics = numTopics, formatted= False, topn = 10):
        i += 1
        print 'Topic #' + str(i) + ":",
        for p, id in topic:
            print dict[int(id)],

        print ""

    #other printing option
        for i in range(0, lda.num_topics-1):
            print lda.print_topic(i)



def main():

    x = split_by_tags()
    n = get_NERs(x)
    print(n)
    text = (n.keys()) #text should be list of entities from NER dictionary to be used for LDA
    print(text)
    extract_topics(text, numTopics=5)


main()