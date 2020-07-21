import unittest
from typing import Any, Dict, List, Tuple

import torch.nn as nn

import utils


class CountParametersTests(unittest.TestCase):

    def test_simple(self):
        net: nn.Module = nn.Linear(2, 2, bias=True)
        num_params: int = utils.count_parameters(net)

        self.assertIsInstance(num_params, int)
        self.assertEqual(num_params, 6)


class CollateFnTests(unittest.TestCase):

    def test_simple(self):
        batch: List[Tuple[str, int]] = [('a', 1)]
        ret = utils.collate_fn(batch)
        self.assertEqual(ret, (('a', ), (1, )))


class DotDictTests(unittest.TestCase):

    def test_simple(self):
        dic: Dict[str, Any] = {
            'string': 'aaa',
            'integer': 12,
            'dict': {'key': 'value'},
            'list': [1, 2]
        }
        dotdict = utils.DotDict(dic)
        self.assertEqual(dotdict.string, dic['string'])
        self.assertEqual(dotdict.integer, dic['integer'])
        self.assertEqual(dotdict.dict.key, dic['dict']['key'])
        self.assertEqual(dotdict.list, dic['list'])

    def test_init_nested_dict(self):
        nested = {
            'key': {'k': {'a': 'b'}}
        }
        dotdict = utils.DotDict(nested)
        self.assertIsInstance(dotdict.key.k, utils.DotDict)

    def test_todict(self):
        dic: Dict[str, Any] = {
            'int': 1,
            'nested': {'key': {'key': 'val'}}
        }
        dotdict = utils.DotDict(dic)
        ret: Dict[Any, Any] = dotdict.todict()

        self.assertDictEqual(ret, dic)
