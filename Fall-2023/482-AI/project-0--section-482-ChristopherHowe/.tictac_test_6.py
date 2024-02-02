#!/usr/bin/env python3
from tictac import TicTacToe
import unittest
import numpy as np


class TestTicTacToe(unittest.TestCase):
    def test_init_board(self):
        ttt = TicTacToe()
        self.assertEqual(ttt.board.shape, (3,3))

    def test_basic_play_game_6(self):
        testcase = -1*np.array([[ 0,0,1],
                             [-1,0,0],
                             [ 1,0,0]])
        player_first = -1
        
        expected = -1*np.array([[ 0,0,1],
                             [-1,1,0],
                             [ 1,0,0]])
        expected_winner = -1
       
        ttt = TicTacToe(testcase, player_first)
        b,p = ttt.play_game()
        print("\nInput board = \n{}".format(testcase))
        print("Input first mover: player {}".format(player_first))
        print("expected output board = \n{}".format(expected))
        print("expected winner: player {}".format(expected_winner))
        print("Your output board = \n{}".format(b))
        print("Your output winner: player {}".format(p))
        
        self.assertEqual(p, expected_winner)
        self.assertEqual(b.tolist(), expected.tolist())

unittest.main()