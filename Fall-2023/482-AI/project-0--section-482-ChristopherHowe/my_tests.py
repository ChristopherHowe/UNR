from tictac import TicTacToe
import unittest
import numpy as np
from tictac import minimax

class TestTicTacToe(unittest.TestCase):
    def test_init_board(self):
        ttt = TicTacToe()
        self.assertEqual(ttt.board.shape, (3,3))

    def test_basic_play_game_1(self):
        testcase = np.array([[ 1,1,0],
                             [-1,1,0],
                             [-1,-1,-1]])
        player_first = 1


        ttt = TicTacToe(testcase, player_first)
        winner = ttt.eval_win()
        self.assertEqual(winner, -1)

class TestMinimax(unittest.TestCase):
    def test_minmax_1(self):
        testcase = np.array([[ 1,1,0],
                             [-1,1,0],
                             [-1,-1,-1]])
        player_first = 1
        ttt = TicTacToe(testcase, player_first)
		minimax_score, minimax_depth = minimax(player_first, ttt.board, ttt.eval_win, ttt.board_full, 0)
        winner = ttt.eval_win()
        self.assertEqual(winner, -1)

unittest.main()
