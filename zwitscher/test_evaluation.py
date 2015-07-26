__author__ = 'arkadi'


def test_overlap_bool():
    from evaluation import overlap_bool
    ground_truth = [[(0, 1), (0, 2)], [(1, 1), (1, 2)]]
    predictions = [[(0, 1), (0, 3)], [(1, 0)]]
    assert overlap_bool(ground_truth, predictions) == 0.5


def test_overlap_f1():
    from evaluation import overlap_f1
    ground_truth = [[(0,1), (0,2)], [(1,1), (1,2)]]
    predictions = [[(0,1), (0,3)], [(1,0)]]
    assert overlap_f1(ground_truth, predictions) == 1.0 / 3

if __name__ == '__main__':
    test_overlap_bool()
    test_overlap_f1()