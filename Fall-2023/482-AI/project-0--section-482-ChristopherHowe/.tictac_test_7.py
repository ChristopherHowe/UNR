#!/usr/bin/env python3
from tictac import TicTacToe
import unittest
import numpy as np


class TestTicTacToe(unittest.TestCase):
    def test_init_board(self):
        ttt = TicTacToe()
        self.assertEqual(ttt.board.shape, (3,3))

    def test_play_game_1(self):
        testcase = np.array([[ 0,0,0],
                             [-1,1,0],
                             [-1,0,0]])
        player_first = 1
        
        expected_1 = np.array([[ 1, -1,  1],
                               [-1,  1, -1],
                               [-1,  1, -1]
                               ])
        expected_2 = np.array([[ 1, -1, -1],
                               [-1,  1,  1],
                               [-1,  1, -1]])
        expected_winner = 0 
       
        ttt = TicTacToe(testcase, player_first)
        b,p = ttt.play_game()
        print("\nInput board = \n{}".format(testcase))
        print("Input first mover: player {}".format(player_first))
        print("expected output boards = \n{}".format(expected_1))
        print("\n or \n{}".format(expected_2))
        print("expected winner: player {}".format(expected_winner))
        print("Your output board = \n{}".format(b))
        print("Your output winner: player {}".format(p))
        
        correct = np.array_equal(b, expected_1) or \
                  np.array_equal(b, expected_2) 
        self.assertEqual(p, expected_winner)
        self.assertTrue(correct)

unittest.main()