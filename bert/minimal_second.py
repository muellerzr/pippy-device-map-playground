from pippy.IR import Pipe, PipeSplitWrapper, annotate_split_points, MultiUseParameterConfig, pipe_split
from pippy.PipelineStage import PipelineStage
import torch 

class ExampleCode(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mm_param = torch.nn.Parameter(torch.randn(512, 512))
        self.mm_param2 = torch.nn.Parameter(torch.randn(512, 512))
        self.lin = torch.nn.Linear(512, 512)
        self.register_buffer("buffer", 0.001 * torch.randn(512))

    def forward(self, x):
        x = torch.mm(x, self.mm_param)
        skip_connection = x
        x = torch.relu(x)
        pipe_split()
        x = torch.mm(x, self.mm_param) + self.buffer
        x = self.lin(x)
        pipe_split()
        x = torch.relu(x)
        x = x + skip_connection
        x = torch.mm(x, self.mm_param2)
        x = self.lin(x)
        return x
    
ec = ExampleCode()
    
ec_pipe = Pipe.from_tracing(ec, MultiUseParameterConfig.TRANSMIT)

old_named_params = zip(*list(ec.named_parameters()))
old_names = list(old_named_params)[0]

for new_name, _ in ec_pipe.named_parameters():
    old_name = ec_pipe.remap_qualname(new_name)
    assert old_name in old_names