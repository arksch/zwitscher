# -*- coding: utf-8 -*-

import re

from HTMLParser import HTMLParser

from connectives import LexConnective, Lexicon

__author__ = 'arkadi'

def parse(dimlex_xml):
    matches = re.findall('<orths.*?</disambi>', dimlex_xml, re.DOTALL)
    lexicon = Lexicon()
    for match in matches:
        lex_conn = LexConnective()
        # Parsing the disambiguous statement
        disambig = re.findall('<conn_d.*?</conn_d>', match)
        assert len(disambig) == 1, \
            'There should be mentioned exactly once whether the connective is disambiguous:\n%s' % match
        disambig = disambig[0]
        disambig = int(re.sub('[^0-9]', '', disambig))
        assert disambig in [0, 1], 'True or false'
        lex_conn.disambi = not bool(disambig)

        # Parsing the different orthographies
        orths = re.findall('<orth.*?</orth>', match, re.DOTALL)
        for orth in orths:
            # Parsing the words of the connective
            parts = []
            parts_xml = re.findall('<part.*?</part>', orth, re.DOTALL)
            for part in parts_xml:
                # Delete all xml brackets. This might still be phrasal.
                parts.append(re.sub('<.*?>', '', part))
                # if 'type="single"' in part:
                #elif 'type="phrasal"' in part:
                #else:
                #    raise ValueError('Could not read part type! Neither single nor phrasal for: "%s"' % part)
            lexicon_key = '_'.join(parts)
            lex_conn.orths.append(lexicon_key)
            # Parsing continuity (not needed since implicit in the number of parts)
            #if 'type="cont"' in orth:
            #    cont = True
            #elif 'type="discont"' in orth:
            #    cont = False
            #else:
            #    raise ValueError('Could not read orth type! Neither cont nor discont for: "%s"' % orth)

            # Parsing canonicalness if we do not already have the canonical key
            if not lex_conn.canonical:
                if 'canonical="1"' in orth:
                    lex_conn.canonical = lexicon_key
        if lex_conn.canonical:
            # Add the connective to the lexicon
            lexicon.connectives[lex_conn.canonical] = lex_conn
            # Update the orthography variants to find the main reading
            lexicon.orthography_variants.update(dict([(variant, lex_conn.canonical) for variant in lex_conn.orths]))
        else:
            raise ValueError('Each connective should have a cannonical orthography!')
    return lexicon

def load(dimlex_path='../data/dimlex.xml'):
    with open(dimlex_path, 'r') as f:
        dimlex_xml = HTMLParser().unescape(unicode(f.read(), 'utf-8')).encode('utf-8')
    return parse(dimlex_xml)

if __name__ == '__main__':
    with open('../data/dimlex.xml', 'r') as f:
        dimlex_xml = HTMLParser().unescape(unicode(f.read(), 'utf-8')).encode('utf-8')
    lexicon = parse(dimlex_xml)
    print lexicon.orthography_variants
    import ipdb; ipdb.set_trace()