#!/usr/bin/env python3
from car import MountainCar
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
        env_id = 'MountainCar-v0' 
        mountain_car = MountainCar(env_id, False, True, 'car.npy')
        all_states = mountain_car.run()
        max_ = np.max(all_states, axis=0)
        result = max_[0] > 0.5
        
        print("Your highest attained position = {}".format(max_[0]))
        print("Position threshold for success >= {}".format(0.5))
        
        self.assertEqual(result,True)

unittest.main()