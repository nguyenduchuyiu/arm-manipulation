import torch
from unified_geometry.models.grounder import SCALE
from unified_geometry.geometry.transforms import sample_se3, apply
from unified_geometry.losses.emd import emd


@torch.inference_mode()
def equivariance(model, P, F, G, index):
    model.eval()
    h, v = model.encode(P, F, index)
    base = model.all_tasks(P, F, index)
    generator = torch.Generator().manual_seed(17501)
    records = []
    for _ in range(4):
        R, t = sample_se3(len(P), generator, device="cpu")
        R = R.cuda()
        t = t.cuda()
        moved = apply(P, R, t)
        hh, vv = model.encode(moved, F, index)
        expected_v = torch.einsum("bncj,bij->bnci", v, R)
        encoder_scalar = float((hh - h).abs().max())
        encoder_vector = float((vv - expected_v).norm(dim=-1).max() * SCALE * 1000)
        decoded = model.decode(
            moved, h, expected_v, torch.zeros(len(P), device="cuda", dtype=torch.long)
        )
        original = model.decode(P, h, v, torch.zeros(len(P), device="cuda", dtype=torch.long))
        decoder_error = float((decoded - apply(original, R, t)).norm(dim=-1).max() * 1000)
        output = model.all_tasks(moved, F, index)
        expected = apply(base, R, t)
        error = (output - expected).norm(dim=-1) * 1000
        assert (
            encoder_scalar < 0.002
            and encoder_vector < 0.03
            and (decoder_error < 0.03)
            and (float(error.max()) < 0.05)
        )
        torch.testing.assert_close(
            emd(output.flatten(0, 1), apply(G, R, t).flatten(0, 1)),
            emd(base.flatten(0, 1), G.flatten(0, 1)),
            rtol=0.0002,
            atol=2e-05,
        )
        records.append(
            dict(
                encoder_scalar_max=encoder_scalar,
                encoder_vector_max_mm=encoder_vector,
                decoder_max_mm=decoder_error,
                end_to_end_max_mm=float(error.max()),
                end_to_end_RMS_mm=float(error.square().mean().sqrt()),
            )
        )
    order = torch.randperm(P.shape[1], device="cuda")
    inverse = torch.empty_like(order)
    inverse[order] = torch.arange(len(order), device="cuda")
    reordered = model.all_tasks(P[:, order], F[:, order], inverse[index[:, order]])
    permutation = float((reordered - base).norm(dim=-1).max() * 1000)
    assert permutation < 0.05
    tasks = torch.arange(4, device="cuda").repeat(len(P))

    def expand(value):
        return (
            value[:, None].expand(len(P), 4, *value.shape[1:]).reshape(len(P) * 4, *value.shape[1:])
        )

    explicit = model(expand(P), expand(F), tasks, expand(index))
    torch.testing.assert_close(explicit, base.flatten(0, 1), rtol=0.0003, atol=2e-05)
    assert not torch.allclose(base[:, 0], base[:, 1])
    return dict(transforms=records, point_permutation_max_mm=permutation)
