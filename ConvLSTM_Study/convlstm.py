import torch.nn as nn
import torch


class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        # input_dim是输入特征的维数，如果是图片数据就是通道数channel 如果是NLP就是词向量的编码长度
        # hidden_dim是每个num_layer的神经元个数
        # kernel_size是卷积核的尺寸
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
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        # padding的目的是保持卷积之后大小不变
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim, # 卷积输入尺寸是 input_dim + hidden_dim
                              out_channels=4 * self.hidden_dim, # out_channels是四倍的hidden_dim因为后续要分成四个门
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        # input_tensor的输入是(batch_size, channel, height, width)
        # 每个cur_state的尺寸是(batch_size, hidden_dim, height, weight) 是调用init_hidden的结果
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis
        # 这个cat按照第二维拼起来 而input_tensor第二维是channel h_cur的第二维是hidden_dim
        # 拼起来是channel+hidden_dim 正好对应conv2d的input_dim+hidden_dim(因为channel就是input_dim)

        combined_conv = self.conv(combined)
        # conv2d的输入是(batch_size, in_channels, height, width)
        # conv2d的输出是(batch_size, out_channels, height_out, width_out)
        # 有了padding之后height = height_out width = width_out (?)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        # 对第二维 c_out->4*hidden_dim维 按照hidden_dim进行分割 分成四份 为四个门
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        # 对应公式计算出四个值

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        # 计算下一个c_n h_n
        print('h_next,c_next维度是：')
        print(h_next.shape)
        print(c_next.shape)
        return h_next, c_next
        # c_n h_n的维度是(batch_size, hidden_dim, height, width)

    def init_hidden(self, batch_size, image_size):
        # 初始化c_n和h_n
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


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
        # 检查kernel的尺寸是否合适

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        # kernel_size和hidden_dim都是list 而且它们的长度都和num_layers相等
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
        # return_all_layers的作用是 True:返回所有的h False:返回h_n 最后一个隐层

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]
            # 判断i是否是0 如果是0 就表示是第一层 那么将input_dim作为input_dim参数
            # 否则说明是非第一层的后面层，他们的input_dim都是上一层的隐层个数

            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)
        # cell_list将num_layers个模型存起来，如num_layers = 3 那么cell_list = [convLSTMCell0, convLSTMCell1, convLSTMCell2]
        # 使用的时候就用cell_list[0], cell_list[1], ... cell_list[num_layers]

    def forward(self, input_tensor, hidden_state=None):
        # print('input_tensor_dim:',input_tensor.shape)
        # input_tensor的维度是(time_stamp, batch_size, channel, height, width) (time_stamp就是seq_len)
        # 如果batch_first = True, 那么就是(batch_size, time_stamp, channel, height, width)
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
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)
        # 如果没有batch_first 那就处理成batch_first的情况

        b, _, _, h, w = input_tensor.size()

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(batch_size=b,
                                             image_size=(h, w))
            # 调用convLSTM init_hidden方法 实际上是调用每个num_layers对应convLSTMCell 的 init_hidden方法
            # 返回的结果有num_layers个 hidden

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1) # 得到sequence_len
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):
            # print('in layer %d the tensor shape:' % (layer_idx))
            # print(cur_layer_input.shape)

            h, c = hidden_state[layer_idx]
            # 首先取第num_layers层的初始化hidden 和 cell
            output_inner = []
            for t in range(seq_len):
                # 传播seq_len个time_stamp
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                # 使用cell_list中当前num_layers的convLSTMCell进行计算 得到h 和 c
                output_inner.append(h)
                # output_inner维度是[seq_len(time)]的 list 每个元素的维度是 (batch_size, 这一层的hidden_dim, height, width)
                # 没问题就是convLSTMCell的h_n的维度
            layer_output = torch.stack(output_inner, dim=1)
            # stack之后 layer_output的维度是(batch_size, seq_len, 这一层的hidden_dim, height, width)
            # 等于在dim=1的位置加了一维 长度是list的长度 也就是seq_len
            cur_layer_input = layer_output
            # 把上一次的五维输出向量作为下一层的输入(cur_layer_input)

            layer_output_list.append(layer_output)
            # 在list中添加layer_output 存放的是每一层的layer_output
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]
        # 如果return_all_layers = False 只返回最后一层的output

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
