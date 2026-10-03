"""Substitution/leakage detection on independently authored toy fixtures."""

import unittest
from evaluation.a35_input_similarity import (code_shape,graph_shape,
                                             lexical_similarity,benchmark_python)


class InputSimilarityTests(unittest.TestCase):
    def test_terminal_oracle_capture_cannot_hide_benchmark_substitution(self):
        benchmark = "a=[1,2]; b=a.copy(); b.append(3)"
        authored = "items=[7,8]; saved=items.copy(); saved.append(9); result=items"
        self.assertEqual(code_shape(benchmark), code_shape(authored))

    def test_numeric_and_identifier_substitutions_keep_python_shape(self):
        a="result=[x*2 for x in [1,4,9] if x%2==0]"
        b="result=[n*7 for n in [6,2,5,8] if n%3==0]"
        self.assertEqual(code_shape(a),code_shape(b))

    def test_computation_changes_after_equal_input_literals_are_not_discarded(self):
        a="values=[1,2]\nresult=sum(values)"
        b="items=[8,9]\nresult=len(items)"
        self.assertNotEqual(code_shape(a),code_shape(b))

    def test_graph_names_cannot_hide_isomorphic_training_examples(self):
        a={"a":[],"b":[],"c":["a","b"],"d":["c"]}
        b={"q":[],"r":[],"s":["q","r"],"t":["s"]}
        cycle={"q":[],"r":["q"],"s":["t"],"t":["s"]}
        self.assertEqual(graph_shape(a),graph_shape(b))
        self.assertNotEqual(graph_shape(a),graph_shape(cycle))

    def test_validation_extra_assertions_do_not_hide_duplicated_implementation(self):
        a="def f(x):\n    return x+1\nassert f(2)==3"
        b="def g(z):\n    return z+9\nassert g(4)==13\nassert g(8)==17"
        self.assertEqual(code_shape(a,function_only=True),code_shape(b,function_only=True))

    def test_number_substitutions_keep_lexical_template_similarity(self):
        self.assertEqual(lexical_similarity("Calculate 4 plus 8; return an integer.",
                                           "Calculate 12 plus 19; return an integer."),1)

    def test_extracts_python_expression_without_using_expected_answers(self):
        prompt="For Python 3, evaluate [z+1 for z in [3,6] if z>3]. Return the JSON array."
        self.assertEqual(code_shape(benchmark_python(prompt)),
                         code_shape("result=[q+1 for q in [5,8] if q>3]"))


if __name__=="__main__": unittest.main()
