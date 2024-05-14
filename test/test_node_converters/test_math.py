import unittest
import torch
import nobuco

class TestPyTorchToKerasConversion(unittest.TestCase):
    def test_log_converter(self):
        # Define the Sign model inside the test
        class Log(torch.nn.Module):
            def __init__(self):
                super(Log, self).__init__()

            def forward(self, input_tensor):
                return torch.log(input_tensor)

        # Initialize the model and input tensor
        torch_model = Log()
        torch_model.eval()
        input_tensor = torch.randn(1, 10, 20)

        # Convert the model and ensure the HTML trace is saved
        keras_model = nobuco.pytorch_to_keras(
            torch_model,
            args=[input_tensor], kwargs=None,
            inputs_channel_order=nobuco.ChannelOrder.TENSORFLOW,
            outputs_channel_order=nobuco.ChannelOrder.TENSORFLOW,
            save_trace_html=True
        )

        # Read the contents of the trace.html file
        with open('trace.html', 'r', encoding='utf-8') as file:
            trace_html = file.read()

        # Assertions for the content of trace_html
        self.assertNotIn('Max diff', trace_html, "The trace HTML should not contain 'Max diff'")


if __name__ == '__main__':
    unittest.main()
