"""Long training must resume with the same optimizer and random stream."""

import copy
import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phoom_matched_training as long_run


def test_resume_reproduces_next_training_update(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(7)
    model = long_run.model_for("mlp", "cpu")
    emas = {
        r: copy.deepcopy(model).eval().requires_grad_(False) for r in long_run.EMA_RATES
    }
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.0)
    generator = torch.Generator().manual_seed(1000)
    x = torch.randn(64, 256)
    schedule = long_run.cosine_schedule()
    long_run.update_once(model, emas, optimizer, generator, x, schedule)
    path = tmp_path / "resume.pt"
    long_run.save_state(path, model, emas, optimizer, generator, 1, [])
    state = torch.load(path, map_location="cpu", weights_only=False)
    resumed, resumed_emas, resumed_optimizer, resumed_generator = long_run.restore(
        state, "cpu"
    )
    expected = long_run.update_once(model, emas, optimizer, generator, x, schedule)
    actual = long_run.update_once(
        resumed, resumed_emas, resumed_optimizer, resumed_generator, x, schedule
    )
    assert actual == expected
    for a, b in zip(model.parameters(), resumed.parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    for r in long_run.EMA_RATES:
        for a, b in zip(emas[r].parameters(), resumed_emas[r].parameters()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
