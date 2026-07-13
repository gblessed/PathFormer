import unittest
from unittest import mock

import torch

from direct_model_inference_backend import DirectModelInferenceBackend


class DirectModelInferenceBackendTest(unittest.TestCase):
    def test_generate_rejects_non_6d_prompts(self):
        backend = DirectModelInferenceBackend(
            scenario_name="city_0_newyork_3p5_s",
            checkpoint_path="/tmp/mock_checkpoint.pth",
        )

        with self.assertRaises(ValueError):
            backend.generate([[1.0, 2.0, 3.0]])

    @mock.patch("direct_model_inference_backend.generate_paths_no_env_batch")
    @mock.patch("direct_model_inference_backend.os.path.exists", return_value=True)
    @mock.patch("direct_model_inference_backend.torch.load")
    @mock.patch("direct_model_inference_backend.PathDecoder")
    def test_generate_wraps_batched_outputs(
        self,
        mock_path_decoder,
        mock_torch_load,
        _mock_exists,
        mock_generate_batch,
    ):
        mock_model = mock.MagicMock()
        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = mock_model
        mock_path_decoder.return_value = mock_model
        mock_torch_load.return_value = {"model_state_dict": {"fake": torch.tensor(1.0)}}

        mock_generate_batch.return_value = (
            torch.tensor(
                [
                    [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]],
                    [[7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]],
                ],
                dtype=torch.float32,
            ),
            torch.tensor([[0.25], [0.75]], dtype=torch.float32),
            torch.tensor(
                [
                    [[1.0, 0.0, 0.0, 1.0]],
                    [[0.0, 1.0, 1.0, 0.0]],
                ],
                dtype=torch.float32,
            ),
        )

        backend = DirectModelInferenceBackend(
            scenario_name="city_0_newyork_3p5_s",
            checkpoint_path="/tmp/mock_checkpoint.pth",
            device="cpu",
        )

        results = backend.generate(
            [
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                [5.0, 4.0, 3.0, 2.0, 1.0, 0.0],
            ],
            max_generate_steps=9,
        )

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["paths"], [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]])
        self.assertEqual(results[1]["interactions"], [[0.0, 1.0, 1.0, 0.0]])
        self.assertEqual(results[0]["path_count"], 0.25)
        self.assertEqual(results[1]["path_count"], 0.75)
        mock_generate_batch.assert_called_once()
        _, kwargs = mock_generate_batch.call_args
        self.assertEqual(kwargs["max_steps"], 9)


if __name__ == "__main__":
    unittest.main()
