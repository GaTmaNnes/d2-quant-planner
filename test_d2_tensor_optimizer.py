#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test de non-régression — d2_tensor_optimizer.py
================================================
Garantit que CHAQUE profil D2 (ECO / BALANCED / PERFORMANCE) respecte son
gate max_loss : loss <= max_loss_gate.

Ce test protège notamment contre la régression observée où apply_max_loss()
était définie mais jamais appelée (le profil PERFORMANCE affichait loss 2.62
pour un gate 2.5).

Usage:
    python test_d2_tensor_optimizer.py
"""

import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import d2_tensor_optimizer as dto


class TestMaxLossGate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Charge le vrai contexte (profil tensoriel + statique + config 27B),
        # pour tester le pipeline réel et pas des données factices.
        cls.ctx = dto.load_context()

    def _report(self, profile):
        c = self.ctx
        return dto.build_report(
            c["payload"], c["forbidden_bytes"], c["scales_bytes"],
            c["sens"], c["extra"], list(dto.MACHINES), profile,
        )

    def test_every_profile_respects_its_gate(self):
        for profile in dto.PROFILES:
            with self.subTest(profile=profile):
                r = self._report(profile)
                gate = dto.PROFILES[profile]["max_loss"]
                self.assertLessEqual(
                    r["loss"], gate + 1e-9,
                    f"{profile}: loss {r['loss']} depasse le gate {gate}",
                )

    def test_reported_gate_matches_config(self):
        for profile in dto.PROFILES:
            with self.subTest(profile=profile):
                r = self._report(profile)
                self.assertEqual(r["max_loss_gate"], dto.PROFILES[profile]["max_loss"])

    def test_assignment_covers_every_tensor(self):
        c = self.ctx
        for profile in dto.PROFILES:
            with self.subTest(profile=profile):
                r = self._report(profile)
                names = {t["name"] for t in c["payload"]}
                self.assertEqual(set(r["assignment"]), names,
                                 f"{profile}: tensors manquants ou surnumeraires")


if __name__ == "__main__":
    unittest.main(verbosity=2)
