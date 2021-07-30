"""
function ConvLSTMCell and ConvLSTM were slightly modified from existing github repo
https://github.com/ndrplz/ConvLSTM_pytorch. 
Liscence: MIT
"""
import torch.nn as nn
import torch

class fMRICNNencoder(nn.Module):
    """
    first step encoders
    """
    def __init__(self,input_dim, output_dim):
        super(fMRICNNencoder, self).__init__()
        self.encodings = nn.Sequential(*[
            nn.Conv3d(in_channels=input_dim, out_channels=16, kernel_size=7, stride=2),
            nn.BatchNorm3d(num_features=16),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2),
            nn.Conv3d(in_channels=16, out_channels=32, kernel_size=5, stride=2),
            nn.BatchNorm3d(num_features=32),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=32, out_channels=output_dim, kernel_size=3, stride=1),
            nn.BatchNorm3d(num_features=output_dim),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=1)
        ])

    def forward(self, x):
        """
        assumes input of shape:
        (B,T,1,X,Y,Z)
        I'm not sure if setting the number of channels to be 1 works. Maybe not?
        """
        return self.encodings(x)


class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.
        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2, kernel_size[2] // 2
        self.bias = bias

        self.conv = nn.Conv3d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        x, y, z = image_size
        return (torch.zeros(batch_size, self.hidden_dim, x, y, z, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, x, y, z, device=self.conv.weight.device))


class ConvLSTM(nn.Module):
    """
    Parameters:
        input_dim: Number of channels in input
        hidden_dim: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        num_layers: Number of LSTM layers stacked on each other
        batch_first: Whether or not dimension 0 is the batch or not
        bias: Bias or no bias in Convolution
        return_all_layers: Return the list of computations for all layers
        Note: Will do same padding.
    Input:
        A tensor of size B, T, C, H, W or T, B, C, H, W
        Should be changed to B,T,1,X,Y,Z
    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    Example:
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
        >> _, last_states = convlstm(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """
        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful
        Returns
        -------
        last_state_list, layer_output
        """
        #if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
        #    input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, _, _, x, y, z = input_tensor.size()

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(batch_size=b,
                                             image_size=(x,y,z))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param

class fMRICNNdecoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        """
        Connect it to the dimension of the encoder
        """
        super(fMRICNNdecoder, self).__init__()

        self.classifier = nn.Sequential(*[
            #nn.Linear(21600,1024),
            #nn.ReLU(inplace=True),
            #nn.Dropout(0.5),
            nn.Linear(64,32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(32,output_dim)     
        ])

        self.final_activ = nn.Sigmoid()

    def forward(self, x):

        #x = self.decoding(x).view(x.shape[0],-1)
        x = x.view(x.shape[0],-1)
        x = self.classifier(x)

        return (self.final_activ(x))

class featureEncoder(nn.Module):
    """
    encoder layer to incorporate features from the slowfast algorithm
    """
    def __init__(self, input_dim, hidden_dim, output_dim, time_resolution):
        super(featureEncoder, self).__init__()
        self.convLayer1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=time_resolution, stride=time_resolution)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu1 = nn.ReLU(inplace=True)
        self.convLayer2 = nn.Conv1d(hidden_dim, output_dim, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(output_dim)
        self.relu2 = nn.ReLU(inplace=True)    

    def forward(self, x):
        x = self.convLayer1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.convLayer2(x)
        x = self.bn2(x)
        outs = self.relu2(x)

        return outs
        
        
class fMRICNNLSTM(nn.Module):

    def __init__(self, input_dim, intermediate_dim, hidden_dim, output_dim, input_time):
        super(fMRICNNLSTM,self).__init__()
        self.input_time = input_time
        self.encoders1 = nn.ModuleList([fMRICNNencoder(input_dim=input_dim, output_dim=intermediate_dim) for _ in range(input_time)])
        self.LSTMs = ConvLSTM(intermediate_dim, hidden_dim, (3,3,3), 2, True, True, False) # Conv LSTM
        self.decoders1 = nn.ModuleList([fMRICNNdecoder(input_dim=hidden_dim, output_dim=output_dim) for _ in range(input_time)])

    def forward(self, x):
        """
        Assume x is of shape B,T,1,X,Y,Z
        """
        x = torch.cat([torch.unsqueeze(self.encoders1[i](x[:,i,:,:,:,:]),1) for i in range(self.input_time)],1)
        x = self.LSTMs(x)[0][-1]
        x = torch.cat([torch.unsqueeze(self.decoders1[i](x[:,i,:,:,:,:]),1) for i in range(self.input_time)],1)
        return x
        

if __name__ == '__main__':

    #=========TEST BRAIN INPUT
    # x = torch.randn(8, 200, 1, 60, 60, 60)
    # TIME_LENGTH = 200
    # my_model = fMRICNNLSTM(input_dim=1, intermediate_dim=64, hidden_dim=64, output_dim=40, input_time=TIME_LENGTH)#.to(device)
    # layer_outputs = my_model(x)
    # print("sounds good")

    #########TEST FEAT INPUT
    rand1 = torch.randn(100,800,2000)
    myNet = featureEncoder(input_dim=800,  hidden_dim=128, output_dim=40, time_resolution=24)
    outs = myNet(rand1)
    print('finished')

