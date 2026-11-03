"""Protocol and model invariants, including inference-label independence."""

import unittest
import numpy as np
import torch
from torch.nn import functional as F

from mm_data import (audit_splits, filtered_ranks, hard_candidates, reciprocal,
                     split_training, training_filters)
from mm_model import SelectiveBiQUE
from train_mm import correction_loss, utility_targets, restore, state_without_features
from refit_mm import (ACTION_COEFFICIENTS, UtilityPolicy, action_mask, action_scores,
                      expected_regret, policy_features)


class MultimodalTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(12)
        torch.set_num_threads(1)
        self.visual = torch.randn(8, 5)
        self.textual = torch.randn(8, 7)
        self.visual[0] = 0
        self.textual[0] = 0
        self.model = SelectiveBiQUE((8, 4, 8), 2, 2, self.visual, self.textual)
        self.queries = torch.tensor([[0, 0, 2], [1, 1, 3], [2, 0, 4]])

    def test_missing_modalities_abstain_and_preserve_structure(self):
        self.model.set_stage("router")
        self.model.eval()
        details = self.model.predict(self.queries, True)
        self.assertTrue(torch.equal(details["weights"][0], torch.tensor([1., 0., 0.])))
        self.assertTrue(torch.equal(details["scores"][0], details["structural"][0]))
        self.assertTrue(torch.equal(details["deltas"][:, :, 0], torch.zeros(3, 2)))

    def test_true_tail_cannot_change_inference(self):
        self.model.set_stage("router")
        self.model.eval()
        altered = self.queries.clone()
        altered[:, 2] = (altered[:, 2] + 1) % 8
        a, b = self.model.predict(self.queries, True), self.model.predict(altered, True)
        self.assertTrue(torch.equal(a["scores"], b["scores"]))
        self.assertTrue(torch.equal(a["weights"], b["weights"]))

    def test_stage_freezing_and_finite_expert_gradients(self):
        self.model.set_stage("experts")
        self.model.train()
        result = self.model.predict(self.queries, True)
        loss = F.cross_entropy(result["structural"] + result["deltas"][:, 0], self.queries[:, 2])
        loss.backward()
        self.assertTrue(all(p.grad is None for p in self.model.backbone.parameters()))
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in self.model.experts.parameters()))
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in self.model.parameters()))
        self.model.set_stage("router")
        self.assertFalse(self.model.experts.training)

    def test_split_before_reciprocals(self):
        rows = np.array([[0, 0, 1], [1, 0, 2], [3, 1, 4], [5, 1, 6]])
        fit, held = split_training(rows, 0.25, 7)
        fit_aug, held_aug = reciprocal(fit, 2), reciprocal(held, 2)
        self.assertFalse(set(map(tuple, fit_aug)) & set(map(tuple, held_aug)))
        self.assertTrue(np.array_equal(held, split_training(rows, 0.25, 7)[1]))

    def test_hard_negatives_exclude_all_training_positives(self):
        q = torch.tensor([[0, 0, 2]])
        filters = {(0, 0): [1, 2]}
        scores = torch.tensor([[1., 10., 9., 8.]])
        indices, valid = hard_candidates(scores, q, filters, 4)
        self.assertEqual(indices[valid].tolist(), [3, 0])
        self.assertEqual(scores[0, 1].item(), 10.)
        self.assertEqual(filtered_ranks(scores, q, filters).item(), 1)
        loss = correction_loss(scores, q, indices, valid, .5)
        self.assertTrue(torch.isfinite(loss))

    def test_gain_labels_and_no_gain_abstention(self):
        q = torch.tensor([[0, 0, 1]])
        details = {"structural": torch.tensor([[3., 2., 0.]]),
                   "deltas": torch.tensor([[[0., 2., 0.], [0., -1., 0.]]]),
                   "available": torch.tensor([[True, True]])}
        labels, utility = utility_targets(details, q, {(0, 0): [1]}, .02, .002)
        self.assertEqual(labels.argmax(-1).item(), 1)
        self.assertGreater(utility[0, 1].item(), 0)
        details["deltas"].zero_()
        labels, _ = utility_targets(details, q, {(0, 0): [1]}, .02, .002)
        self.assertTrue(torch.equal(labels, torch.tensor([[1., 0., 0.]])))

    def test_gain_minus_harm_is_mrr_change(self):
        baseline = torch.tensor([1., .5, .2])
        fused = torch.tensor([.5, 1., .25])
        gain = (fused - baseline).clamp_min(0).mean()
        harm = (baseline - fused).clamp_min(0).mean()
        self.assertAlmostEqual((gain - harm).item(), (fused.mean() - baseline.mean()).item(), places=6)

    def test_checkpoint_roundtrip_and_split_audit(self):
        before = self.model.backbone.entity_embeddings.weight.detach().clone()
        state = state_without_features(self.model)
        with torch.no_grad():
            self.model.backbone.entity_embeddings.weight.zero_()
        restore(self.model, state)
        self.assertTrue(torch.equal(before, self.model.backbone.entity_embeddings.weight))
        audit = audit_splits({"train": np.array([[0, 0, 1]]),
                              "valid": np.array([[0, 0, 1]]), "test": np.array([[2, 0, 3]])})
        self.assertFalse(audit["clean"])
        self.assertEqual(audit["train_valid"], 1)

    def test_discrete_actions_match_single_corrections_and_missing_masks(self):
        self.model.set_stage("router")
        self.model.eval()
        details = self.model.predict(self.queries, True)
        coefficients = torch.tensor(ACTION_COEFFICIENTS)
        scores = action_scores(details, coefficients)
        self.assertTrue(torch.equal(scores[:, 0], details["structural"]))
        self.assertTrue(torch.equal(scores[:, 2], details["structural"] + details["deltas"][:, 1]))
        mask = action_mask(details["available"], coefficients)
        self.assertEqual(mask[0].tolist(), [True, False, False, False])
        features = policy_features(details)
        self.assertFalse(features.requires_grad)
        self.assertEqual(tuple(features.shape), (3, 16))

    def test_utility_regret_prefers_actual_gain_and_ignores_invalid_actions(self):
        utilities = torch.tensor([[0., .1, -.1, 100.]])
        available = torch.tensor([[True, True, True, False]])
        good = torch.tensor([[-20., 20., -20., -1e9]], requires_grad=True)
        bad = torch.tensor([[20., -20., -20., -1e9]], requires_grad=True)
        self.assertLess(expected_regret(good, utilities, available).item(), 1e-5)
        loss = expected_regret(bad, utilities, available)
        self.assertAlmostEqual(loss.item(), .1, places=5)
        loss.backward()
        self.assertTrue(torch.isfinite(bad.grad).all())

    def test_utility_policy_missing_features_force_null_action(self):
        policy = UtilityPolicy(4)
        features = torch.randn(3, 16)
        policy.normalize_from(features)
        available = torch.tensor([[True, False, False, False]] * 3)
        logits = policy(features, self.queries[:, 1], available)
        self.assertTrue(torch.equal(logits.softmax(-1)[:, 0], torch.ones(3)))


if __name__ == "__main__":
    unittest.main()
