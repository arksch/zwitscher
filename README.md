# zwitscher - German Discourse Parsing

This package is a part of a project at the University of Potsdam. The goal was
to parse Twitter discourse in a PDTB style. This part of the project assumes
as an input a constituency tree including POS-tags and tokens, together with
the sentence-nested positions of the connectives themselves. From this input
the script *pipeline.py* labels the argument spans that belong to the connective.

The ideas for the discourse parser are along the lines of Lin. et. al.
https://www.comp.nus.edu.sg/~nght/pubs/nle13.pdf

For more information about the implementatin read the pdf in the article folder

At the current state it is possible to parse from end to end,
but the results are still rather poor. It is expected, that they can still
be improved with a better feature set.

## How to use it out of the box?

### Setup

To get all the data and packages you need run

   ```sh
   git clone https://github.com/arksch/zwitscher.git
   cd zwitscher
   pip install -r requirements.txt
   ```

Test your setup by running the main script with default parameters

   ```sh
   cd zwitscher
   python pipeline.py
   ```

### Run the script

The script has the following options

  ```sh
  $ python pipeline.py --help
Usage: pipeline.py [OPTIONS]

  Running a pipeline to label argument spans in a syntax annotated discourse
  with known positions of connectives

Options:
  -s, --syntax_trees_xml_path TEXT
                                  Path to a file with syntactic information in
                                  TigerXML format
  -c, --connective_positions_path TEXT
                                  Path to a file with nested connective
                                  positions in json format
                                  E.g. '[[[14, 0],
                                  [14, 5]], [[11, 0]], [[2, 0], [2, 1]]]'
  -sc, --sent_dist_classification_path TEXT
                                  Path to the sentence distance classifier
                                  trained with the learn_sentdist.py script
  -as, --argspan_classification_path TEXT
                                  Path to the argspan classifier trained with
                                  the learn_argspan.py script
  --help                          Show this message and exit.
  ```

### Prepare your own data

A discourse is typically a paragraph, but it can be of arbitrary size.
If you want to parse other discourses, you will need TigerXML Syntax trees
for each of them. Store each in a file. An example can be found in

   zwitscher/data/test_pcc/syntax/maz-00001.xml

Since this parser assumes, that discourse connective tokens have already been identified,
you will need to store this information as well for each discourse.
Take care, that the tokenization matches the syntax trees.
The positions of the discourse connectives are saved in a JSON file with sentencewise-nested positions.
I.e. a list of connectives where each connective consists of token positions and each position
is a sentence index with a token index in the given sentence.
An example matching the syntax example above can be found in

   zwitscher/data/test_pcc/maz-00001_nested_conn_indices.json
   
   [
   [[14, 0], [14, 5]],
   [[11, 0]],
   [[2, 0], [2, 1]],
   [[2, 22]],
   [[4, 5]],
   [[3, 2]],
   [[8, 10]],
   [[9, 0]],
   [[8, 14]]
   ]

To create these indices from PCC files you can run the script

   ```sh
   python connective_positions_from_pcc.py -d data/test_pcc/connectors/maz-00001.xml -s data/test_pcc/syntax/maz-00001.xml
[[[14, 0], [14, 5]], [[11, 0]], [[2, 0], [2, 1]], [[2, 22]], [[4, 5]], [[3, 2]], [[8, 10]], [[9, 0]], [[8, 14]]]

   python connective_positions_from_pcc.py       
[[[14, 0], [14, 5]], [[11, 0]], [[2, 0], [2, 1]], [[2, 22]], [[4, 5]], [[3, 2]], [[8, 10]], [[9, 0]], [[8, 14]]]

   python connective_positions_from_pcc.py --help
Usage: connective_positions_from_pcc.py [OPTIONS]

  A script to print nested or unnested indices of connectives in the
  sentences of a discourse from PCC files

Options:
  -d, --discourse_path TEXT  Path to the discourse xml file. Defaults to
                             data/test_pcc/connectors/maz-00001.xml
  -s, --syntax_path TEXT     Path to the syntax xml file. Defaults to
                             data/test_pcc/syntax/maz-00001.xml
  -n, --nested               Output the nested or unnested positions
  --help                     Show this message and exit.
   ```

A fellow project is developing a tool to extract discourse tokens from 
Twitter text.

This project uses the DiMLex, a lexicon of German discourse connectives, created by:

Manfred Stede. "DiMLex: A Lexical Approach to Discourse Markers" In: A. Lenci, V. Di Tomaso (eds.): Exploring the Lexicon - Theory and Computation. Alessandria (Italy): Edizioni dell'Orso, 2002.

You can download a current public version of the DiMLex at 
https://github.com/discourse-lab/dimlex
to replace the one copied to this repository.

### Training new classifiers

You might want to train new classifiers.
There are two scripts that also evaluate their results of the algorithm.
You can start the scripts from the zwitscher/zwitscher folder with

   ```sh
   $ python learn_sentdist.py
   ...similar output as below...
   $ python learn_argspan.py
   
   Learning classifier for finding the main nodes of connectors.
   =============================================================
   Loading data...
   Unpickling gold data from data/PCC_disc.pickle
   Loaded data
   Cleaning data...
   ...
   Training classifiers for arg0 labeling
   ======================================
   Majority baseline: 0.881298
   Cross validating Logistic regression classifier...
   Cross validated Logistic Regression classifier
   scores: [ 0.91257996  0.89434365  0.92102455  0.90918803  0.90277778]
   mean score: 0.907983
   
   Training classifiers for arg1 labeling
   ======================================
   Majority baseline: 0.885781
   Cross validating Logistic regression classifier...
   Cross validated Logistic Regression classifier
   scores: [ 0.88473853  0.88580576  0.88580576  0.88580576  0.88568376]
   mean score: 0.885568
   Learning classifiers on the whole data set...
   Learned classifier on the whole data set
   ...done
   Pickling sent_dist classifier to ed0f4620cd1c4402bfa582182be83b44_argspan_classification_dict.pickle
   ```

You can see the options of the scripts below. When you run them,
they will create a new classification set for either sentence distance or
argument span, that can be read by the pipeline.py script.

The evaluation of the scripts is printed to standard output. For this a classifiers is first
trained on a part of the data and evaluated on the rest. The classifier that is saved is then
trained on the complete data set.

If you want to use your own evaluation methods you will have to embed them inside the script.
Currently no end-to-end evaluation is implemented.
If you want to evaluate on a held out data set you will also have to implement that method.

By default both learning scripts load the pickled PCC. If you want to load
new PCC data or you have changed some methods of the underlying objects,
then you will have to change the unpickle_gold option to False.

The scripts have the following options:

   ```sh
   $ python learn_argspan.py --help
Usage: learn_argspan.py [OPTIONS]

  Learning and storing classifiers for argument spans of discourse
  connectivesThe output file can be passed used by the pipeline with python
  pipeline.py -as 123uuid_argspan_classification_dict.pickle

Options:
  -f, --feature_list TEXT       A comma separated list of features. By default
                                connective_lexical,nr_of_siblings,path_to_node
                                ,node_cat
  -lf, --label_features TEXT    A comma separated list of features that are
                                labels. By defaultconnective_lexical,path_to_n
                                ode,node_cat
  -cf, --connector_folder TEXT  The folder to find the connectors from the
                                potsdam commentary corpusCan be left empty if
                                unpickle_gold is True
  -pf, --pickle_folder TEXT     The folder for all the pickles
  -uf, --unpickle_features      Unpickle precalculated features. Useful for
                                dev
  -ug, --unpickle_gold          Unpickle gold connector data. Useful for dev
  -pc, --pickle_classifier      Pickle the classifier for later use, e.g. in
                                the pipeline.Note that this adds an uuid to
                                the output, so it doesnt overwrite former
                                outputOtherwise this script will just print
                                the evaluation
  --help                        Show this message and exit.
   ```


## Performance

The sentence distance classification reaches an accuracy of over 90%, which
is a reasonable result over the baseline of 57% majority labeling with both
arguments in the same sentence. Note, that this does not yet use syntactic 
features.

On the contrary, the argument span labeling only reaches an f-score of 23% for
the internal argument span and 25% for the external argument span.
Both values were calculated by counting words that were labeled as in the
ground truth as a true positive, and other labeled words as false
positive/negative respectively.

## Development

Obviously, this package is still far from perfect. Feel free to contribute!

### Tests

You can run tests with

  ```sh
  py.test
  ```

### Possible Improvements

- Better features (features.py)
  - Syntactic features for sentence distance
  - Use a dictionary for the words used in argument span labeling features
  - Use capitalization
  - Not only use the first connective word for argument span features
  - Faster feature calculation
- Better evaluation
  - Nested cross-validation, as the current evaluation is biased by selection of
the best classifier
  - Evaluate the full pipeline, not only the sentence distance and argument span
labeling separately
  - Wrap the argument span evaluation into a sklearn metric,
  so that it can be used directly in the classification training
  - More meaningful metrics, that allow to evaluate what actually goes wrong
  and are comparable to Lin et al (partial match of nouns and verbs, perfect
  match)
  - Do not count punctuation
  - Calculate some baselines and upper bounds
     - Full sentence labeling for different sentences
     - Labeling with the gold standard argument nodes
  - Evaluate the labeling for argument spans in different sentences
- Try other ways to create argument spans from the argument nodes
(Lin et al use tree subtraction, but that might not be a good choice for PCC)
- Try other classification algorithms
  - Don't train the classifier on other cases than previous and same sentence
- Create command line arguments for the classifier training scripts
- Improve the parsing or fix inconsistencies in the PCC
- Fix one of the many ToDos in the code and refactor it to be readable




