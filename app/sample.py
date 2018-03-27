#!/usr/bin/env python
# -*- coding: utf-8 -*-
#=========================================================================
#
# Copyright (c) 2017 <> All Rights Reserved
#
#
# Author: Hai Liang Wang
# Date: 2017-11-14:14:01:11
# http://programmerts.blogspot.com/2014/08/determine-question-type-sentence-using.html
#=========================================================================

"""

http://www.zmonster.me/2016/06/08/use-stanford-nlp-package-in-nltk.html
https://stackoverflow.com/questions/15003136/cfg-using-pos-tags-in-nltk
https://stackoverflow.com/questions/17695611/nltk-context-free-grammar-genaration
https://stackoverflow.com/questions/3522372/how-to-config-nltk-data-directory-from-code
https://stackoverflow.com/questions/23429117/saving-nltk-drawn-parse-tree-to-image-files
http://programmerts.blogspot.com/2014/08/determine-question-type-sentence-using.html

## API
* grammar: 
https://www.nltk.org/api/nltk.html#module-nltk.grammar

* tree: 
https://www.nltk.org/api/nltk.html#nltk.tree.Tree

* RecursiveDescentParser: 
https://www.nltk.org/api/nltk.parse.html#module-nltk.parse.recursivedescent
"""
from __future__ import print_function
from __future__ import division

__copyright__ = "Copyright (c) 2017 . All Rights Reserved"
__author__ = "Hai Liang Wang"
__date__ = "2017-11-14:14:01:11"


import os
import sys
curdir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(curdir, os.path.pardir))

if sys.version_info[0] < 3:
    reload(sys)
    sys.setdefaultencoding("utf-8")
    # raise "Must be using Python 3"
else:
    unicode = str

import subprocess

'''
Config nltk
'''
os.environ["NLTK_DATA"] = os.path.join(
    curdir, os.path.pardir, "data", "nltk_data")

# Get ENV
ENVIRON = os.environ.copy()

import nltk
nltk.download('punkt')


'''
Utilities
'''
def exec_cmd(cmd):
    p = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True,
        env=ENVIRON)
    out, err = p.communicate()
    return out, err


'''
Save tree to png file.
'''
from nltk.tree import Tree
from nltk.draw import TreeWidget
from nltk.draw.tree import TreeView
from nltk.draw.util import CanvasFrame
from recursive_descent_parser_model import RecursiveDescentParser

def save_tree_png(tree, ouput):
    '''
    将Tree保存为png, 不支持中文字符
    '''
    print("save_tree_png", tree)
    t = Tree.fromstring(tree)
    ps = "%s.ps" % ouput
    TreeView(t)._cframe.print_to_file(ps)
    print(">> Generate Tree Image [%s], tree string [%s] ..." % (ouput, tree))
    exec_cmd("convert %s %s" % (ps, ouput))


import unittest


class Test(unittest.TestCase):
    '''

    '''

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_sample(self):
        print("test_sample")
        # This is a CFG grammar, where:
        # Start Symbol : S
        # Nonterminal : NP,VP,DT,NN,VB
        # Terminal : "I", "a" ,"saw" ,"dog"
        grammar = nltk.grammar.CFG.fromstring("""
            S -> NP VP
            NP -> DT NN | NN
            VP -> VB NP
            DT -> "a"
            NN -> "I" | "dog"
            VB -> "saw"
        """)
        sentence = "I saw a dog".split()
        parser = RecursiveDescentParser(grammar)
        final_tree = parser.parse(sentence)

        for i in final_tree:
            print(i)

    def test_nltk_cfg_qtype(self):
        print("test_nltk_cfg_qtype")
        gfile = os.path.join(
            curdir,
            os.path.pardir,
            "config",
            "grammar.question-type.cfg")
        question_grammar = nltk.data.load('file:%s' % gfile)

        def get_missing_words(grammar, tokens):
            """
            Find list of missing tokens not covered by grammar
            """
            missing = [tok for tok in tokens
                       if not grammar._lexical_index.get(tok)]
            return missing

        # sentence = "do i need code when i dep now ?"
        sentence = "what is your name"

        # check type
        sent = sentence.split()
        missing = get_missing_words(question_grammar, sent)
        target = []
        for x in sent:
            if x in missing:
                continue
            target.append(x)

        rd_parser = RecursiveDescentParser(question_grammar)
        result = []
        print("target: ", target)
        for tree in rd_parser.parse(target):
            result.append(x)
            print("Question Type\n", tree)

        if len(result) == 0:
            print("Not Question Type")

    def test_nltk_cfg_en(self):
        print("test_nltk_cfg_en")
        grammar = nltk.CFG.fromstring("""
         S -> NP VP
         VP -> V NP | V NP PP
         V -> "saw" | "ate"
         NP -> "John" | "Mary" | "Bob" | Det N | Det N PP
         Det -> "a" | "an" | "the" | "my"
         N -> "dog" | "cat" | "cookie" | "park"
         PP -> P NP
         P -> "in" | "on" | "by" | "with"
         """)

        # Make your POS sentence into a list of tokens.
        sent = "Mary saw Bob".split()

        # Load the grammar into the RecursiveDescentParser.
        rd_parser = RecursiveDescentParser(grammar)

        result = []

        for i, tree in enumerate(rd_parser.parse(sent)):
            result.append(tree)
            save_tree_png(
                str(tree),
                os.path.join(
                    curdir,
                    os.path.pardir,
                    "tmp",
                    "nltk_cfg_en_%s.png" %
                    i))

        assert len(result) > 0, "Can not recognize CFG tree."

        print(result)


    def test_nltk_cfg_zh(self):
        print("test_nltk_cfg_zh")
        grammar = nltk.CFG.fromstring("""
         S -> N VP
         VP -> V NP | V NP | V N
         V -> "尊敬"
         N -> "我们" | "老师"
         """)

        # Make your POS sentence into a list of tokens.
        sent = "我们 尊敬 老师".split()

        # Load the grammar into the RecursiveDescentParser.
        rd_parser = RecursiveDescentParser(grammar)

        result = []

        for i, tree in enumerate(rd_parser.parse(sent)):
            result.append(tree)
            print("Tree [%s]: %s" % (i + 1, tree))

        assert len(result) > 0, "Can not recognize CFG tree."
        if len(result) == 1 :
            print("Draw tree with Display ...")
            result[0].draw()
        else:
            print("WARN: Get more then one trees.")

        print(result)

def test():
    unittest.main()


if __name__ == '__main__':
    test()
