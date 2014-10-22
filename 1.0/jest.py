'''
various routines to parse the annotated infinite jest
'''
import re
import pdb
import os
import operator 
import pickle
import itertools

####
# PLOT PREDICTIONS OVER ACTUAL ### BRILLIANT!
####

###
# for plotting
import pylab
from pylab import *
from matplotlib.patches import Rectangle

import nltk

THRESHOLD = .33
#THRESHOLD = .25
PATH_TO_JEST = "infinite_jest_annotated.txt"
print "THRESHOLD IS %s" % THRESHOLD
# from mallet
PATH_TO_NER_TOPIC_KEYS = "NER_topic_keys"
#PATH_TO_NER_TOPIC_KEYS = "NER_topic_keys4"
PATH_TO_FULL_TEXT_TOPIC_KEYS = "full_text_topic_keys"
#PATH_TO_FULL_TEXT_TOPIC_KEYS = "ft_topic_keys4"

# map the topic numbers to narratives
# this has to be hand-coded, i'm afraid
# *make sure this is the case prior to evaluation*
# i.e., you don't want to evaluate with this mapping
# if, e.g., '0' is better representative of <EHDRDH>

## TMP
#tops_to_nars = {"0":"<AFR>", "1":"<EHDRH>", "2":"<ETA>"}
tops_to_nars = {"0":"<AFR>", "1":"<EHDRH>", "2":"<ETA>"}

#tops_to_nars = {"0":"<ETA>", "1":"<INC>", "2":"<AFR>", "3":"<EHDRH>"}

def get_all_tags():
    jest = open(PATH_TO_JEST).read()
    p = re.compile(r'</.*?>') # finds the *closing* tags
    all_tags = p.findall(jest)
    all_tags = [x.replace("</", "<") for x in list(set(all_tags))]
    return all_tags

def get_tags_in_text(text):
    p = re.compile(r'<.*?>') # finds the *closing* tags
    all_tags = p.findall(text)
    tags = [x for x in list(set(all_tags))]
    # which are close tags?
    return all_tags
    
def get_topic_keys(full_text=False):
    all_topics = None
    if full_text:
        all_topics = open(PATH_TO_FULL_TEXT_TOPIC_KEYS).readlines()
    else:
        # entities
        all_topics = open(PATH_TO_NER_TOPIC_KEYS).readlines()
    topics_to_keys = {}

    # map the topic numbers to narratives
    # this has to be hand-coded, i'm afraid
    for topic in all_topics:
        cur_topic = topic.split("\t")
        topics_to_keys[tops_to_nars[cur_topic[0]]] = [x for x in cur_topic[2].split(" ") if x!="\n"]
    return topics_to_keys

def is_close_tag(tag):
    return tag.startswith("</")
    
def get_all_text_for_tag():
    pass
    
def get_jest():
    return open(PATH_TO_JEST).readlines()


def text_count(seg_text, top_terms_for_nar):
    matches = []
    for t in seg_text:
        if t.lower() in top_terms_for_nar:
            matches.append(t.lower())
    return len(matches), list(set(matches))

def entity_count(entity_d_for_seg, entities_for_narrative):
    total_count = 0
    
    # normalize the dictionary, because mallet outputs lower case
    # entities
    lowered_entity_d = {}
    matched_entities = []
    for ent in entity_d_for_seg.keys():
        lowered_entity_d[ent.lower()] = entity_d_for_seg[ent]

    for entity in entities_for_narrative:
        if entity in lowered_entity_d:
            total_count += lowered_entity_d[entity]
            matched_entities.append(entity)
    return total_count, matched_entities


def meta_infer_narratives(preds_path, exclude_non_top_narrs=False, min_entity_count=None, full_text=False, 
                        gimme_seg_probs=False):
    global tops_to_nars
    narrs = tops_to_nars.values()
    all_perms = [perm for perm in itertools.permutations(narrs)]
    best_micro_f1 = 0
    best_perm, best_results = None, None
    for perm in all_perms:
        tops_to_nars = {"0":perm[0], "1":perm[1], "2":perm[2], "3":perm[3]}
        cur_inference = infer_narratives(preds_path, exclude_non_top_narrs=exclude_non_top_narrs,\
                                            min_entity_count=min_entity_count, full_text=full_text)
        
        if cur_inference['micro_averaged']['f1'] >= best_micro_f1:
            best_micro_f1 = cur_inference['micro_averaged']['f1']
            best_perm = perm
            best_results = cur_inference
            
    return (best_results, best_perm)


###
# this function is poorl named -- we're doing inference and evaluation here!
def infer_narratives(preds_path, exclude_non_top_narrs=False, min_entity_count=None, full_text=False, 
                        gimme_seg_probs=False):
    '''
    preds_path is the path to the topic (narrative) predictions output by mallet.
                *NOTE THAT WE ASSUME* that the 0th topic is best `thought of' 
                as <AFR>, 1 as <EHDRH> and 2 as ETA. this was by visual inspection of the topics,
                but the mapping is actually unnecessary/for convienence of interpretation. we
                could instead simply map the different inferred topics to all possible
                tags and keep the best one. 

    if exclude_non_top_narrs in this case, we will only evaluate the inference procedure on
                    segments (passages) that contain at least one of the three top
                    narrative tags -- others will be ignored.
            
    if min_entity_count is used, we enforce that for a passage to be assigned
                    to a narrative, it must contain at least one entity belonging to
                    said narratives' top 'keys' or representative entities. if full_text
                    is True, then we use the full text topic keys, of course (and in this
                    case the variable name is someting of a misnomer)
                    
    '''
    print "THRESHOLD IS %s" % THRESHOLD

    # grab the tags
    tags_for_segs, segments, d = get_tags_for_segments()

    # if min_entity_count was not specified, default to True for entities
    # and False for full-text
    if min_entity_count is None:
        min_entity_count = False if full_text else True

    ####
    # also read in entities
    ###
    segs_to_entities = None
    if not full_text:
        segs_to_entities = pickle.load(open("segs_to_NEs.pickle"))

    # parse the predictions (from mallet)
    pred_d = parse_mallet_output(preds_path)

    # performance measures
    tps, fps, tns, fns = [], [], [], []

    # i.e., the entities most strongly associated with each narrative
    narratives_to_keys = get_topic_keys(full_text=full_text)
    seg_probs = [] # keep our probabilities
    
    predicted_tags_for_segs = []

    for seg_i, tags in enumerate(tags_for_segs):
        # what did we predict?
        
        # sometime don't want to consider segments with no assigned tags
        if not exclude_non_top_narrs or \
                    any([tag in tags_for_segs[seg_i] for tag in tops_to_nars.values()]):
              
            pred_d_new_keys = {}
            for topic, p in pred_d[str(seg_i)].items():
                pred_d_new_keys[tops_to_nars[topic]] = p
            
            ###
            # here we deal with the following
            #   * when to not assign to *any* narratives?
            #   * when to assign to multiple narratives?
            # 
            #   to do this, we order by probabilities (descending), then walk down 
            #   the narratives making sure they probability meets a reasonable
            #   threshold *and* it contains at least one entity.
            sorted_by_probs = sorted(pred_d_new_keys.iteritems(), \
                                        key=operator.itemgetter(1), reverse=True)
           

            pred_tags = []
            pred_dist = {}

            for predicted_tag, p in sorted_by_probs:
                # setting this higher leads to more liberal predictions (more things)
                # positive. in the case that every narrative can safely assumed to belong
                # to at least one narrative (i.e., skipping non-tagged segments), probably
                # this conditional can be all but ignored. it does help eliminate false positives
                # for untagged segments (ie., those that are not part of any known narrative)
                # if we check that said segment contains at least one entity from the top
                # k
                k = 500
                top_terms_for_predicted_nar = narratives_to_keys[predicted_tag][:k]
                if not full_text:
                    matched_count, matched_ents = \
                        entity_count(segs_to_entities[str(seg_i)], top_terms_for_predicted_nar)
                else: 
                    # then we do plain text matching
                    #pdb.set_trace()
                    matched_count, matched_ents = \
                        text_count(segments[seg_i], top_terms_for_predicted_nar)

                if p <= THRESHOLD or (matched_count==0 and min_entity_count):
                    pass  
                else:
                    pred_tags.append(predicted_tag)
                    pred_dist[predicted_tag] = p

            predicted_tags_for_segs.append(pred_tags)
            # renorm
            total_mass = sum(pred_dist.values())
            if total_mass < 1.0:
                mass_to_add = 1.0 - total_mass 
            
            # distribute the mass proportionally
            num_tags = float(len(pred_tags))
    
            z_wrong = mass_to_add/num_tags

            for tag in pred_dist.keys():
                z = mass_to_add * (pred_dist[tag]/total_mass)
                pred_dist[tag] = pred_dist[tag] + z
                print "z_wrong is %s; correct is: %s" % (z_wrong, z)
                print "z_wrong - z=%s" % (z_wrong - z)


            seg_probs.append(pred_dist)
      
            ####
            #
            # maybe do per-narrative analysis? e.g., sens./prec. for <AFR>,
            # <ETA>, etc. It looks like we do poorly on <AFR> w.r.t.
            # precision when we don't skip, i.e., we tend to default
            #
            #####

            # ok, how did we do for this segment?
            for tag in tops_to_nars.values(): # these are the only tags we care about
                if tag in tags_for_segs[seg_i]:
                    # then this tag *should* have been predicted for this segment
                    if tag in pred_tags:
                        tps.append((segments[seg_i], seg_i, tag))
                    else:
                        fns.append((segments[seg_i], seg_i, tag)) # we missed it
                else:
                    # the tag *should not* be predicted for this segment
                    if tag in pred_tags:
                        fps.append((segments[seg_i], seg_i, tag))
                    else:
                        tns.append((segments[seg_i], seg_i, tag))

    if gimme_seg_probs:
        return seg_probs
    
    return compute_metrics(tags_for_segs, predicted_tags_for_segs)
    #return {"tps":tps, "fps":fps, "tns":tns, "fns":fns}

def compute_metrics(true_tags_for_segs, predicted_tags_for_segs, narratives_we_care_about=["<AFR>", "<EHDRH>","<ETA>"]):#, "<INC>"]):
    ''' note -- i insist on using 'tag' and 'narrative' interchangably '''

    narr_to_metrics_d = {}
    overall_conf_mat = {"tps":0, "tns":0, "fns":0, "fps":0} # for micro averaged performance
    for narr in narratives_we_care_about:
        narr_to_metrics_d[narr] = {"tps":0, "tns":0, "fns":0, "fps":0}

    for seg_i in xrange(len(true_tags_for_segs)):
        pred_tags_for_seg = predicted_tags_for_segs[seg_i]
        for tag in narratives_we_care_about:
            if tag in true_tags_for_segs[seg_i]:
                # then this tag *should* have been predicted for this segment
                if tag in pred_tags_for_seg:
                    narr_to_metrics_d[tag]["tps"]+=1
                    overall_conf_mat["tps"]+=1
                else:
                    # we missed it
                    narr_to_metrics_d[tag]["fns"]+=1
                    overall_conf_mat["fns"]+=1
            else:
                # the tag *should not* be predicted for this segment
                if tag in pred_tags_for_seg:
                    narr_to_metrics_d[tag]["fps"]+=1
                    overall_conf_mat["fps"]+=1
                else:
                    narr_to_metrics_d[tag]["tns"]+=1
                    overall_conf_mat["tns"]+=1
    
    # now compute metrics for each narrative
    for narr in narratives_we_care_about:
        metrics_for_narr = calc_metrics(narr_to_metrics_d[narr])
        #pdb.set_trace()
        for met in metrics_for_narr.keys():
            narr_to_metrics_d[narr][met] = metrics_for_narr[met]
    
    # micro-averaged results
    
    narr_to_metrics_d["micro_averaged"] = calc_metrics(overall_conf_mat)

    # macro averaged
    macro_d = {'precision':0.0, 'recall':0.0, 'f1':0.0}
    for narr in narratives_we_care_about:
        for met in macro_d.keys():
            macro_d[met] += narr_to_metrics_d[narr][met]

    
    narr_to_metrics_d['macro_averaged'] = {}
    for met in ['precision', 'recall', 'f1']:
        narr_to_metrics_d['macro_averaged'][met] = \
                    macro_d[met]/float(len(narratives_we_care_about))
        
    return narr_to_metrics_d

def calc_metrics(two_by_two_d):
    try:
        precision = float(two_by_two_d["tps"])/(float(two_by_two_d["tps"] + two_by_two_d["fps"]))
    except:
        # div by 0, presumabl
        precision = 0.0

    try:
        recall = float(two_by_two_d["tps"])/(float(two_by_two_d["tps"] + float(two_by_two_d["fns"])))
    except:
        recall = 0.0

    if recall == precision == 0.0:
        f1 = 0.0
    else:
        f1 = (2*recall*precision)/ (precision+recall)

    return {"precision":precision, "recall":recall, "f1":f1}

def baseline_infer(strategy="all-same", majority_tag="<INC>", \
                    round_robin_ordering=["<ETA>", "<EHRDH>", "<AFR>", "<INC>"],\
                    skip=False):
    tags_for_segs, segments, d = get_tags_for_segments()
    preds = []

    if skip:
        # filter list of segments here
        pass

    for i,seg in enumerate(segments):
        if strategy == "all-same":
            preds.append(majority_tag)
        elif strategy == "round-robin":
            # cycle through the tags
            pred_i = i % len(round_robin_ordering)
            preds.append(round_robin_ordering[pred_i])
    
    return compute_metrics(tags_for_segs, preds)

def get_best_rr(narrs=["<ETA>", "<EHRDH>", "<AFR>"], skip=False):
    all_perms = [perm for perm in itertools.permutations(narrs)]
    best_f1 = 0.0
    best_results = None
    for perm in all_perms:
        cur_results = baseline_infer(strategy="round-robin", round_robin_ordering=perm, skip=skip)
        print "f1 for perm %s: %s" % (perm, cur_results["micro_averaged"]["f1"] )
        if cur_results["micro_averaged"]["f1"] > best_f1 or best_results is None:
            best_results = cur_results
            best_f1 = cur_results["micro_averaged"]["f1"] 
        
def make_pred_from_dist(pred_dist):
    # naive for now; *only* return one narrative per
    # segment. this needs some thought...
    max_p = 0.0
    pred_tag = None
    for tag, p in pred_dist.items():
        if p > max_p:
            pred_tag = tag
            max_p = p
    return (pred_tag, max_p)



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

def spit_out_text_for_segs(tags_we_care_about=["<EHDRH>", "<AFR>", "<ETA>"]):
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
        
def _remove_bracks(tag):
    return tag.replace("<", "").replace(">", "")
    
   
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

def get_NER_dicts_for_segs(dir_path):
    # read in tags, strip them
    segments = os.listdir(dir_path)
    seg_texts = []
    segs_to_NERs = {}
    all_NERs_to_types = {}
    for seg in segments:
        # map entities to counts (i.e., # of occurences in this seg)
        segs_to_NERs[seg], NERs_to_types = get_NERs(os.path.join(dir_path, seg))
        for NER in NERs_to_types:
            all_NERs_to_types[NER] = NERs_to_types[NER]
    return segs_to_NERs, all_NERs_to_types

def strip_all_tags(seg_text):
    return strip_tags(seg_text, get_tags_in_text(seg_text))


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

def parse_mallet_output(fpath):
    pred_d = {} # map segments to their distributions
    preds = open(fpath).readlines()
    for pred_line in preds:
        split_line = pred_line.split("\t")
        try:
            doc_id =  split_line[1].split("/")[-1] # use the filename
            seg_id = str(int(doc_id))
            seg_id = doc_id
            pred_d[seg_id] = {}
            # the - here drops a tab followed by a new line character
            for i in range(2, len(split_line)-1, 2):
                pred_d[seg_id][split_line[i]] = float(split_line[i+1])
        except:
            print "skipping line %s" % pred_line
        
    return pred_d  


def main_plus_narratives_plot():
    ### with <THE_ENTERTAINMENT>
    colors = {"<EHDRH>":'grey', "<ETA>":'blue', "<AFR>":'purple', "<INC>":'green'}
    tags_for_segs, segs, opened_d = get_tags_for_segments()
    seg_colors, seg_lengths = [], []
    for seg_i in xrange(len(tags_for_segs)):
        colors_for_seg = []
        for tag in tags_for_segs[seg_i]: 
            if tag in colors:
                colors_for_seg.append(colors[tag])
            else:
                # 'background' narratives, I guess
                #colors_for_seg.append("black")
                pass
        seg_colors.append(colors_for_seg)
        seg_len = len(" ".join(segs[seg_i]))
        seg_lengths.append(seg_len)
       
    # now plot them
    clf()
    
    y_scalar = .2
    ylim(0, 2)

    # so the briefest segment will have length 1
    #pdb.set_trace()
    #z = float(min([len(seg) for seg in segs if not len(seg) == 0]))
    z = float(min([seg_len for seg_len in seg_lengths if not seg_len==0]))
    z = z/.25
    # map colors to y coordinates
    ygap = .8
    ybase = .1
    y_indices = {colors["<ETA>"]:.1, colors["<EHDRH>"]:ybase+ygap, colors["<AFR>"]:ybase+2*ygap, colors["<INC>"]:ybase+3*ygap}
    gately_xranges, eta_xranges, marathe_xranges, entertainment_ranges, black_xranges = [],[],[],[], []
    xoffset = 9000 # to make room for text!
    x_index = 0
    for seg_i in xrange(len(seg_colors)):
        colors_to_plot = list(set(seg_colors[seg_i]))
        
        sort(colors_to_plot)
        ymax = len(colors_to_plot)
        #normed_len = float(len(segs[seg_i]))/z
        normed_len = float(seg_lengths[seg_i])/z
        for c_i, color in enumerate(colors_to_plot):
            #y_i = ymax - c_i
            y_i = y_indices[color]*y_scalar
            #y_i = 1.0
 
            '''
            cur_xrange = (seg_i, seg_i+1.0)
            if color == colors["<GATELY>"]:
                gately_xranges.append(cur_xrange)
            elif color == colors["<ETA>"]:
                eta_xranges.append(cur_xrange)
            elif color == colors["<MARATHE>"]:
                marathe_xranges.append(cur_xrange)
            else:
                black_xranges.append(cur_xrange)
            '''
            #axhline(y=y_i*.5, xmin=seg_i/z, xmax=(seg_i+1.0)/z, color=color, linewidth=10, alpha=.5)
            print "plotting color %s at x_index %s with len %s (there are %s colors to plot)" %\
                (color, x_index+xoffset, normed_len, len(colors_to_plot))
                

            rect = Rectangle((xoffset+x_index, y_i), normed_len, 1*y_scalar, facecolor=color, alpha=.5)
            gca().add_patch(rect)
            # bump the current index
        
        x_index += normed_len
        

       #pdb.set_trace()

    xticks([])
    yticks([])
    xlim(0, x_index+xoffset)

    ax=axes()
    

    for x in ["left", "right", "top", "bottom"]:
        ax.spines[x].set_visible(False)
    
   # xlabel(r"Meta-Narrative $\rightarrow$")
    xlabel(r"Passage (Meta-Narrative $\rightarrow$)")
    
    xtext= 0
    ybase_text = .075
    text(xtext, ybase_text, "ETA", color=colors["<ETA>"])
    text(xtext, ybase_text+.175, "EHDRH", color=colors["<EHDRH>"])
    text(xtext, ybase_text+.35, "AFR", color=colors["<AFR>"])
    text(xtext, ybase_text+.52, "INC", color=colors["<INC>"])
    savefig("plot_ent.pdf")
    return seg_colors




def main_narratives_plot():
    colors = {"<EHDRH>":'grey', "<ETA>":'blue', "<AFR>":'purple'}
    tags_for_segs, segs, opened_d = get_tags_for_segments()
    seg_colors, seg_lengths = [], []
    for seg_i in xrange(len(tags_for_segs)):
        colors_for_seg = []
        for tag in tags_for_segs[seg_i]: 
            if tag in colors:
                colors_for_seg.append(colors[tag])
            else:
                # 'background' narratives, I guess
                #colors_for_seg.append("black")
                pass
        seg_colors.append(colors_for_seg)
        seg_len = len(" ".join(segs[seg_i]))
        seg_lengths.append(seg_len)
       
    # now plot them
    clf()
    
    y_scalar = .2
    ylim(0, 2)

    # so the briefest segment will have length 1
    #pdb.set_trace()
    #z = float(min([len(seg) for seg in segs if not len(seg) == 0]))
    z = float(min([seg_len for seg_len in seg_lengths if not seg_len==0]))
    z = z/.25
    # map colors to y coordinates
    ygap = .8
    ybase = .1
    y_indices = {colors["<ETA>"]:.1, colors["<EHDRH>"]:ybase+ygap, colors["<AFR>"]:ybase+2*ygap}
    gately_xranges, eta_xranges, marathe_xranges, black_xranges = [],[],[],[]
    xoffset = 9000 # to make room for text!
    x_index = 0
    for seg_i in xrange(len(seg_colors)):
        colors_to_plot = list(set(seg_colors[seg_i]))
        
        sort(colors_to_plot)
        ymax = len(colors_to_plot)
        #normed_len = float(len(segs[seg_i]))/z
        normed_len = float(seg_lengths[seg_i])/z
        for c_i, color in enumerate(colors_to_plot):
            #y_i = ymax - c_i
            y_i = y_indices[color]*y_scalar
            #y_i = 1.0
 
            '''
            cur_xrange = (seg_i, seg_i+1.0)
            if color == colors["<GATELY>"]:
                gately_xranges.append(cur_xrange)
            elif color == colors["<ETA>"]:
                eta_xranges.append(cur_xrange)
            elif color == colors["<MARATHE>"]:
                marathe_xranges.append(cur_xrange)
            else:
                black_xranges.append(cur_xrange)
            '''
            #axhline(y=y_i*.5, xmin=seg_i/z, xmax=(seg_i+1.0)/z, color=color, linewidth=10, alpha=.5)
            print "plotting color %s at x_index %s with len %s (there are %s colors to plot)" %\
                (color, x_index+xoffset, normed_len, len(colors_to_plot))
                

            rect = Rectangle((xoffset+x_index, y_i), normed_len, 1*y_scalar, facecolor=color, alpha=.5)
            gca().add_patch(rect)
            # bump the current index
        
        x_index += normed_len
        

       #pdb.set_trace()

    xticks([])
    yticks([])
    xlim(0, x_index+xoffset)

    ax=axes()
    

    for x in ["left", "right", "top", "bottom"]:
        ax.spines[x].set_visible(False)
    
   # xlabel(r"Meta-Narrative $\rightarrow$")
    xlabel(r"Passage (Meta-Narrative $\rightarrow$)")
    
    xtext= 0
    ybase_text = .075
    text(xtext, ybase_text, "ETA", color=colors["<ETA>"])
    text(xtext, ybase_text+.175, "EHDRH", color=colors["<EHDRH>"])
    text(xtext, ybase_text+.35, "AFR", color=colors["<AFR>"])
    savefig("plot.pdf")
    return seg_colors


def main_narratives_plot_with_preds(preds):
    colors = {"<EHDRH>":'grey', "<ETA>":'blue', "<AFR>":'purple'}
    tags_for_segs, segs, opened_d = get_tags_for_segments()
    seg_colors, seg_lengths = [], []
    for seg_i in xrange(len(tags_for_segs)):
        colors_for_seg = []
        for tag in tags_for_segs[seg_i]: 
            if tag in colors:
                colors_for_seg.append(colors[tag])
            else:
                # 'background' narratives, I guess
                #colors_for_seg.append("black")
                pass
        seg_colors.append(colors_for_seg)
        seg_len = len(" ".join(segs[seg_i]))
        seg_lengths.append(seg_len)
    # now plot them
    clf()
    
    y_scalar = .2
    ylim(0, 2)

    # so the briefest segment will have length 1
    #pdb.set_trace()
    #z = float(min([len(seg) for seg in segs if not len(seg) == 0]))
    z = float(min([seg_len for seg_len in seg_lengths if not seg_len==0]))
    z = z/.25
    # map colors to y coordinates
    ygap = .8
    ybase = .1
    y_indices = {colors["<ETA>"]:.1, colors["<EHDRH>"]:ybase+ygap, colors["<AFR>"]:ybase+2*ygap}
    gately_xranges, eta_xranges, marathe_xranges, black_xranges = [],[],[],[]

    x_index = 0
    xoffset = 9000
    for seg_i in xrange(len(seg_colors)):
        colors_to_plot = list(set(seg_colors[seg_i]))
        
        sort(colors_to_plot)
        ymax = len(colors_to_plot)
        #normed_len = float(len(segs[seg_i]))/z
        normed_len = float(seg_lengths[seg_i])/z
        ### plot ground truth
        for c_i, color in enumerate(colors_to_plot):
            #y_i = ymax - c_i
            y_i = y_indices[color]*y_scalar
            #y_i = 1.0
 
            #axhline(y=y_i*.5, xmin=seg_i/z, xmax=(seg_i+1.0)/z, color=color, linewidth=10, alpha=.5)
            print "plotting color %s at x_index %s with len %s (there are %s colors to plot)" %\
                (color, x_index+xoffset, normed_len, len(colors_to_plot))
                

            rect = Rectangle((x_index+xoffset, y_i), normed_len, 1*y_scalar, facecolor="white", alpha=1.0)

            gca().add_patch(rect)
            # bump the current index
        
        hatch = None

        #### now plot predictions
        edgecolor = None
        for tag in preds[seg_i]:
            hatch = None
            color = colors[tag]
            y_i = y_indices[color]*y_scalar
            alpha = .5
            if not color in colors_to_plot:
                #color = "white"
                hatch='x'
                #alpha = .5
                edgecolor= color
                color = 'white'
            

            #alpha=alpha*preds[seg_i][tag]
            rect = Rectangle((x_index+xoffset, y_i), normed_len, 1*y_scalar, facecolor=color,
                                          edgecolor=edgecolor, hatch=hatch, alpha=alpha)
            gca().add_patch(rect)

        x_index += normed_len
        

       #pdb.set_trace()

    xticks([])
    yticks([])
    xlim(0, x_index+xoffset)

    ax=axes()
    

    for x in ["left", "right", "top", "bottom"]:
        ax.spines[x].set_visible(False)
    
   # xlabel(r"Meta-Narrative $\rightarrow$")
    xlabel(r"Passage (Meta-Narrative $\rightarrow$)")
    pdb.set_trace()
    xtext=0
    ybase_text = .075
    text(xtext, ybase_text, "ETA", color=colors["<ETA>"])
    text(xtext, ybase_text+.175, "EHDRH", color=colors["<EHDRH>"])
    text(xtext, ybase_text+.35, "AFR", color=colors["<AFR>"])
    savefig("plot_preds.pdf")
    return seg_colors



