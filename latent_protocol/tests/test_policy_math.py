"""Run in the GPU environment (also works on CPU with a working PyTorch install)."""
import itertools
import unittest

try:
    import torch
    import torch.nn.functional as F
except (ImportError, OSError, ValueError):
    torch = None

from latent_protocol.core import leave_one_out_advantages


@unittest.skipIf(torch is None, 'A working PyTorch installation is needed')
class PolicyMathTest(unittest.TestCase):
    def test_leave_one_out_score_gradient_matches_exact_expected_reward(self):
        # Exhaust all two-sample Bernoulli groups; baseline must not bias gradient.
        theta = torch.tensor(.3, dtype=torch.float64, requires_grad=True)
        p = theta.sigmoid()
        expected = 2 * p
        exact, = torch.autograd.grad(expected, theta)
        gradient = 0.
        for actions in itertools.product([0, 1], repeat=2):
            theta = torch.tensor(.3, dtype=torch.float64, requires_grad=True)
            dist = torch.distributions.Bernoulli(logits=theta)
            action = torch.tensor(actions, dtype=torch.float64)
            logp = dist.log_prob(action)
            probability = float(logp.detach().sum().exp())
            adv = torch.tensor(leave_one_out_advantages([2*a for a in actions]))
            objective = (adv * logp).mean()
            grad, = torch.autograd.grad(objective, theta)
            gradient += probability * float(grad)
        self.assertAlmostEqual(gradient, float(exact), places=6)

    def test_causal_shift_and_action_mask(self):
        # Prompt 3 tokens, actions 2: prediction logits at positions 2 and 3.
        logits = torch.randn(1, 5, 7)
        kept = logits[:, -3:, :][0, :-1]
        self.assertTrue(torch.equal(kept, logits[0, 2:4]))
        kept[:, [5, 6]] = -float('inf')
        targets = torch.tensor([2, 4])
        score = -F.cross_entropy(kept, targets, reduction='sum')
        direct = kept.log_softmax(-1)[torch.arange(2), targets].sum()
        self.assertTrue(torch.allclose(score, direct))


if __name__ == '__main__': unittest.main()
