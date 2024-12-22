import onnxruntime
import time

class ONNXBackend():
    def __init__(self, model_path: str, device: str = "cpu") -> None:    
        device = "cpu"
        if device == "cuda":
            providers = ["CUDAExecutionProvider"]
        elif device == "cpu":
            providers = ["CPUExecutionProvider"]                
        self.ort_session = onnxruntime.InferenceSession(model_path, providers=providers)

    def infer(self, inputs, logger):
        logger.info("Got inference request!")
        
        outputs = self.ort_session.get_outputs()
        output_names = list(map(lambda output: output.name, outputs))
        input_names = self.ort_session.get_inputs()
        ort_inputs = {}
        t_start = time.time()

        # Loop over the different inputs and as them as the model input
        for i in range(0, len(inputs)):
            ort_inputs[input_names[i].name] = inputs[i]

        output = self.ort_session.run(output_names, ort_inputs)        
        t_end = time.time()
        logger.info(f"Inference session run: {t_end - t_start:.3f}")

        return output
