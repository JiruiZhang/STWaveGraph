from pathlib import Path
import importlib.util
import json
import unittest
import torch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('wave_model_test', ROOT/'STWaveGraph/model.py')
model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model)


def response_errors(dtype=torch.float64):
    errors = []
    for alpha in [0.001, 0.05, 0.2, 1.0, 2.0]:
        for beta in [0.01, 1.0, 5.0]:
            for time in [0.001, 0.1, 1.0, 3.0]:
                layer = model.SpectralGraphWaveLayer(
                    2, n_bands=1, alpha_init=[alpha],
                    beta_init=[beta], time_init=[time]).to(dtype=dtype)
                a = layer._positive(layer.raw_alpha).clamp(1e-3, 2.0)[0]
                b = layer._positive(layer.raw_beta).clamp(1e-3, 5.0)[0]
                t = layer._positive(layer.raw_time).clamp(1e-3, 3.0)[0]
                critical = (a.square()/4/b).detach()
                lam = torch.stack([
                    critical*0, critical*0.5, critical*(1-1e-8),
                    critical, critical*(1+1e-8), critical*1.5,
                    critical*0+1, critical*0+2])
                u = torch.zeros((len(lam), 2), dtype=dtype)
                v = torch.zeros_like(u)
                u[:, 0] = 1
                v[:, 1] = 1
                observed = layer(u, v, torch.eye(len(lam), dtype=dtype), lam)
                generator = torch.zeros((len(lam), 2, 2), dtype=dtype)
                generator[:, 0, 1] = 1
                generator[:, 1, 0] = -b*lam
                generator[:, 1, 1] = -a
                reference = torch.matrix_exp(generator*t)[:, 0, :]
                errors.append((observed-reference).abs().max().item())
    return {'dtype': str(dtype), 'parameter_sets': len(errors),
            'scalar_mode_cases': len(errors)*8,
            'max_absolute_error': max(errors)}


class WaveSolverTests(unittest.TestCase):
    def test_response_matches_matrix_exponential(self):
        result = response_errors()
        print('RESPONSE_DOUBLE', json.dumps(result))
        self.assertLess(result['max_absolute_error'], 1e-9)

    def test_float32_response(self):
        result = response_errors(torch.float32)
        print('RESPONSE_FLOAT', json.dumps(result))
        self.assertLess(result['max_absolute_error'], 2e-5)

    def test_critical_and_adjacent_gradients(self):
        for offset in [0.0, -1e-10, 1e-10, -1e-3, 1e-3]:
            inputs = (
                torch.tensor([0.25+offset], dtype=torch.float64, requires_grad=True),
                torch.tensor(1.0, dtype=torch.float64, requires_grad=True),
                torch.tensor(1.0, dtype=torch.float64, requires_grad=True),
                torch.tensor(1.0, dtype=torch.float64, requires_grad=True))
            self.assertTrue(torch.autograd.gradcheck(
                model.damped_wave_transfer, inputs, eps=1e-6, atol=2e-6, rtol=1e-4))

    def test_initial_conditions_and_zero_mode(self):
        lam = torch.tensor([0.0, 0.25, 2.0], dtype=torch.float64)
        a = torch.tensor(1.0, dtype=torch.float64)
        b = torch.tensor(1.0, dtype=torch.float64)
        h, v = model.damped_wave_transfer(lam, a, b, torch.tensor(0.0))
        torch.testing.assert_close(h, torch.ones_like(lam), atol=1e-12, rtol=0)
        torch.testing.assert_close(v, torch.zeros_like(lam), atol=1e-12, rtol=0)
        t = torch.tensor(1.2, dtype=torch.float64)
        h, v = model.damped_wave_transfer(lam[:1], a, b, t)
        torch.testing.assert_close(h, torch.ones_like(h), atol=1e-12, rtol=0)
        torch.testing.assert_close(v, (-torch.expm1(-a*t)/a).reshape(1), atol=1e-12, rtol=0)

    def test_series_transition(self):
        lam = torch.tensor([0.25-1.000001e-4, 0.25-0.999999e-4,
                            0.25+0.999999e-4, 0.25+1.000001e-4],
                           dtype=torch.float64)
        a = torch.tensor(1.0, dtype=torch.float64)
        b = torch.tensor(1.0, dtype=torch.float64)
        t = torch.tensor(1.0, dtype=torch.float64)
        h, v = model.damped_wave_transfer(lam, a, b, t)
        generator = torch.zeros((len(lam), 2, 2), dtype=torch.float64)
        generator[:, 0, 1] = 1
        generator[:, 1, 0] = -lam
        generator[:, 1, 1] = -1
        reference = torch.matrix_exp(generator)[:, 0, :]
        torch.testing.assert_close(torch.stack([h, v], 1), reference, atol=1e-12, rtol=1e-11)

    def test_multiband_projection_and_complement(self):
        torch.manual_seed(31)
        u = torch.randn(9, 2, dtype=torch.float64)
        v = torch.randn_like(u)
        basis = torch.linalg.qr(torch.randn(9, 6, dtype=torch.float64)).Q
        lam = torch.tensor([0, 0.03, 0.25, 0.4, 1.5, 2], dtype=torch.float64)
        layer = model.SpectralGraphWaveLayer(2, n_bands=3).double()
        observed = layer(u, v, basis, lam)
        expected_modes = []
        for k in range(6):
            band = k//2
            alpha = layer._positive(layer.raw_alpha).clamp(1e-3, 2.0)[band]
            beta = layer._positive(layer.raw_beta).clamp(1e-3, 5.0)[band]
            time = layer._positive(layer.raw_time).clamp(1e-3, 3.0)[band]
            generator = torch.stack([torch.stack([alpha*0, alpha*0+1]),
                                     torch.stack([-beta*lam[k], -alpha])])
            gains = torch.matrix_exp(generator*time)[0]
            expected_modes.append(gains[0]*(basis[:, k]@u)+gains[1]*(basis[:, k]@v))
        rho = layer._positive(layer.raw_res_scale).clamp(0, 2)
        expected = basis@torch.stack(expected_modes)+rho*(u-basis@(basis.T@u))
        torch.testing.assert_close(observed, expected, atol=1e-10, rtol=1e-10)

    def test_network_forward_backward(self):
        torch.manual_seed(42)
        net = model.WaveGraphSTNet(5, 8, 3, proj_dim=4, dropout=0.0)
        x = torch.randn(8, 5)
        adj = torch.eye(8)
        basis = torch.eye(8)
        lam = torch.linspace(0, 2, 8)
        outputs = net(x, x+0.01, adj, basis, lam)
        loss = sum(value.square().mean() for value in outputs)
        loss.backward()
        for name, p in net.named_parameters():
            self.assertIsNotNone(p.grad, name)
            self.assertTrue(torch.isfinite(p.grad).all().item(), name)
        with torch.no_grad():
            for p in net.parameters(): p.add_(p.grad, alpha=-1e-4)
        self.assertTrue(all(torch.isfinite(value).all().item() for value in outputs))


if __name__ == '__main__':
    unittest.main(verbosity=2)
