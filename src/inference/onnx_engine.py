"""ONNX export and CPU-optimized inference engine.

Key insight: For single-sample inference, CPU + ONNX Runtime is FASTER
than GPU due to eliminated kernel launch and memory transfer overhead.
Typical: 0.5-2ms on CPU vs 3-8ms on GPU for models this size.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger


class ONNXExporter:
    """Export PyTorch models to ONNX format for production inference."""

    @staticmethod
    def export(
        model,
        input_shapes: dict[str, tuple],
        output_path: str,
        opset_version: int = 17,
        dynamic_axes: Optional[dict] = None,
    ) -> str:
        """Export a PyTorch model to ONNX.

        Args:
            model: PyTorch model (nn.Module).
            input_shapes: Dict of input name → shape, e.g. {"x": (1, 120, 6)}.
            output_path: Where to save the .onnx file.
            opset_version: ONNX opset version.
            dynamic_axes: Dynamic axes specification.

        Returns:
            Path to exported ONNX file.
        """
        import torch

        model.eval()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Create dummy inputs
        dummy_inputs = {}
        for name, shape in input_shapes.items():
            dummy_inputs[name] = torch.randn(*shape)

        # Handle single vs multiple inputs
        if len(dummy_inputs) == 1:
            dummy_input = list(dummy_inputs.values())[0]
            input_names = list(dummy_inputs.keys())
        else:
            dummy_input = tuple(dummy_inputs.values())
            input_names = list(dummy_inputs.keys())

        if dynamic_axes is None:
            dynamic_axes = {name: {0: "batch_size"} for name in input_names}
            dynamic_axes["output"] = {0: "batch_size"}

        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            input_names=input_names,
            output_names=["output"],
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
            do_constant_folding=True,
        )

        # Verify
        import onnx

        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)

        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        logger.info(f"ONNX model exported: {output_path} ({file_size_mb:.1f} MB)")
        return str(output_path)


class ONNXInferenceEngine:
    """CPU-optimized ONNX Runtime inference engine.

    Designed for low-latency single-sample inference at tick speed.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        n_threads: int = 4,
        enable_profiling: bool = False,
    ):
        import onnxruntime as ort

        self.model_path = model_path

        # Configure session options for low latency
        session_opts = ort.SessionOptions()
        session_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_opts.intra_op_num_threads = n_threads
        session_opts.inter_op_num_threads = 1  # single-sample, no parallelism needed
        session_opts.enable_mem_pattern = True
        session_opts.enable_cpu_mem_arena = True

        if enable_profiling:
            session_opts.enable_profiling = True

        # Select provider
        if device == "cpu":
            providers = ["CPUExecutionProvider"]
        elif device == "cuda":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

        self.session = ort.InferenceSession(
            model_path, session_opts, providers=providers
        )

        # Cache input/output names
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]

        # Warm up
        self._warmup()
        logger.info(
            f"ONNX engine loaded: {model_path} | "
            f"device={device} | threads={n_threads} | "
            f"inputs={self.input_names} | outputs={self.output_names}"
        )

    def _warmup(self, n_runs: int = 10) -> None:
        """Warm up the inference engine to stabilize latency."""
        input_shapes = {
            inp.name: [
                d if isinstance(d, int) else 1
                for d in inp.shape
            ]
            for inp in self.session.get_inputs()
        }

        dummy_feeds = {
            name: np.random.randn(*shape).astype(np.float32)
            for name, shape in input_shapes.items()
        }

        for _ in range(n_runs):
            self.session.run(self.output_names, dummy_feeds)

    def predict(self, **inputs: np.ndarray) -> np.ndarray:
        """Run inference.

        Args:
            **inputs: Named numpy arrays matching model input names.

        Returns:
            Model output as numpy array.
        """
        feed = {}
        for name in self.input_names:
            if name not in inputs:
                raise ValueError(f"Missing input: {name}. Expected: {self.input_names}")
            arr = inputs[name]
            if arr.dtype != np.float32:
                arr = arr.astype(np.float32)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            feed[name] = arr

        outputs = self.session.run(self.output_names, feed)
        return outputs[0]

    def predict_timed(self, **inputs: np.ndarray) -> tuple[np.ndarray, float]:
        """Run inference and return (output, latency_ms)."""
        start = time.perf_counter()
        output = self.predict(**inputs)
        latency_ms = (time.perf_counter() - start) * 1000
        return output, latency_ms

    def predict_action(self, **inputs: np.ndarray) -> tuple[int, float, float]:
        """Run inference and return (action, confidence, latency_ms).

        Returns:
            action: Argmax of output logits.
            confidence: Softmax probability of selected action.
            latency_ms: Inference time.
        """
        output, latency_ms = self.predict_timed(**inputs)

        # Softmax
        exp_out = np.exp(output - output.max(axis=-1, keepdims=True))
        probs = exp_out / exp_out.sum(axis=-1, keepdims=True)

        action = int(probs[0].argmax())
        confidence = float(probs[0, action])
        return action, confidence, latency_ms

    def benchmark(self, n_runs: int = 1000, **sample_inputs: np.ndarray) -> dict:
        """Benchmark inference latency.

        Returns:
            Dict with mean, std, p50, p95, p99 latencies in ms.
        """
        latencies = []
        for _ in range(n_runs):
            _, latency = self.predict_timed(**sample_inputs)
            latencies.append(latency)

        latencies = np.array(latencies)
        results = {
            "mean_ms": float(latencies.mean()),
            "std_ms": float(latencies.std()),
            "p50_ms": float(np.percentile(latencies, 50)),
            "p95_ms": float(np.percentile(latencies, 95)),
            "p99_ms": float(np.percentile(latencies, 99)),
            "min_ms": float(latencies.min()),
            "max_ms": float(latencies.max()),
            "n_runs": n_runs,
        }
        logger.info(
            f"Benchmark ({n_runs} runs): "
            f"mean={results['mean_ms']:.2f}ms, "
            f"p95={results['p95_ms']:.2f}ms, "
            f"p99={results['p99_ms']:.2f}ms"
        )
        return results
