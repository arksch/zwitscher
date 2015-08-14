# zwitscher - German Discourse Parsing

This package is a part of a project at the University of Potsdam. The goal was
to parse Twitter discourse in a PDTB style. This part of the project assumes
as an input a constituency tree including POS-tags and tokens, together with
the sentence-nested positions of the connectives themselves. From this input
the script *pipeline.py* labels the argument spans that belong to the connective.

The ideas for the discourse parser are along the lines of Lin. et. al.
https://www.comp.nus.edu.sg/~nght/pubs/nle13.pdf

At the current state it is possible to parse from end to end,
but the results are still rather poor. It is expected, that they can still
be improved with a better feature set.

## How to use it out of the box?

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

If you want to parse other discourses, you will need TigerXML Syntax trees
for each of them.
A discourse is typically a paragraph, but it can be of arbitrary size.
You also need a JSON file in which you save the sentence-nested positions of
the argument spans. An example can be found in

zwitscher/data/test_pcc/maz-00001_nested_conn_indices.json

A fellow project is developing a tool to extract discourse tokens from 
Twitter text.

The script has the following options

  ```sh
  python pipeline.py --help
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


### Training new classifiers

Once you are done improving the scripts in any way, you might want to train
new classifiers. There are two scripts that also evaluate the given state of 
the algorithm.

The two scripts learn_argspan.py and learn_sentdist.py do not have command line
options yet. You will have to modify them from within (also to tell them
which features they shall use). When you run them,
they will create a new classification set for either sentence distance or
argument span, that can be read by the pipeline.py script.

By default both learning scripts load the pickled PCC. If you want to load
new PCC data or you have changed some methods of the underlying objects,
then you will have to change the unpickle_gold option to False.

