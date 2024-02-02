#!/usr/bin/env python3
from cart import CartPole
import unittest
import numpy as np


class TestTicTacToe(unittest.TestCase):
    # def test_init_board(self):
    #     ttt = TicTacToe3D()
    #     # brd,winner = ttt.play_game()
    #     self.assertEqual(ttt.board.shape, (3,3,3))

    def test_1(self):
        player_first = 1
        expected_winner = 1
        env_id = 'CartPole-v1' 
        cartpole = CartPole(env_id, False, True, 'cart.npy')
        all_states = cartpole.run()
        max_ = np.max(all_states, axis=0)
        print("max = {}".format(max_))
        result_1 = max_[0] <= 2.4
        result_2 = max_[2] <= 0.226893
        result = result_1 and result_2
        print("Your max cart position = {}".format(max_[0]))
        print("Your max pole angle = {}".format(max_[2]))
        print("Cart position for success <= {}".format(2.4))
        print("Pole angle for success <= {} radians".format(0.226893))
        
        self.assertEqual(result,True)

unittest.main()