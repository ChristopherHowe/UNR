#!/usr/bin/env python3
from tictac3d import TicTacToe3D
import unittest
import numpy as np


class TestTicTacToe(unittest.TestCase):
    # def test_init_board(self):
    #     ttt = TicTacToe3D()
    #     # brd,winner = ttt.play_game()
    #     self.assertEqual(ttt.board.shape, (3,3,3))

    def test_basic_play_game_3(self):
        testcase = np.array([[[ 0, 0, 0],
                              [ 0, 1, 0],
                              [ 0, 0,-1]],
                             
                             [[ 0, 0, 0],
                              [ 1, 0, 0],
                              [ 0, 0, 0]],
                             
                             [[ 0, 1, 0],
                              [ 0,-1,-1],
                              [ 0, 1, 0]]])
        player_first = -1
        
        expected_winner = -1
       
        ttt = TicTacToe3D(board=testcase.copy(), player=player_first)
        b,p = ttt.play_game()

        print("\nInput board = \n{}".format(testcase))
        print("Input first mover: player {}".format(player_first))
        print("expected winner: player {}".format(expected_winner))
        print("Your output winner: player {}".format(p))
        
        self.assertEqual(p, expected_winner)

unittest.main()