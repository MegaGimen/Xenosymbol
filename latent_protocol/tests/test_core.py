import unittest
from latent_protocol.core import content_tokens, reward, leave_one_out_advantages, selected_positions


class DefinitionsTest(unittest.TestCase):
    def test_last_content_position_excludes_eos(self):
        content, stopped = content_tokens([4, 9, 2, 0], {2})
        self.assertEqual(content, [4, 9]); self.assertTrue(stopped)
        self.assertEqual(selected_positions(10, len(content)), (9, 11))

    def test_truncation_is_not_a_stop(self):
        self.assertEqual(content_tokens([4, 9], {2}), ([4, 9], False))

    def test_reward_is_exact_user_objective(self):
        self.assertAlmostEqual(reward(.3, 10, .005), -.35)

    def test_leave_one_out_baseline_excludes_current_sample(self):
        self.assertEqual(leave_one_out_advantages([1, 2, 3]), [-1.5, 0, 1.5])
        self.assertEqual(leave_one_out_advantages([2, 2, 2]), [0, 0, 0])

    def test_empty_response_has_no_fabricated_last_token(self):
        with self.assertRaises(ValueError): selected_positions(3, 0)


if __name__ == '__main__': unittest.main()
