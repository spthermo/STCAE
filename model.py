import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.init import xavier_uniform_, zeros_

def conv(in_channels, out_channels, kernel, pad, dil):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel, stride=1, padding=pad),
        nn.ELU(True)
    )

def conv_downsample(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
        nn.ELU(True)
    )

def conv_seg(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.Sigmoid()
    )

def conv_1x1(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0),
        nn.ELU(True)
    )


class Interpolate(nn.Module):
    def __init__(self, size, mode):
        super(Interpolate, self).__init__()
        """
        Args:
            size: expected size after interpolation
            mode: interpolation type (e.g. bilinear, nearest)
        """
        self.interp = nn.functional.interpolate
        self.size = size
        self.mode = mode
        
    def forward(self, x):
        out = self.interp(x, size=self.size, mode=self.mode, align_corners=False)
        
        return out


class Encoder(nn.Module):
    def __init__(self, ndf, dil):
        super(Encoder, self).__init__()

        self.ndf = ndf
        self.dil = dil

        self.conv1 = conv(3, self.ndf, kernel=7, pad=3, dil=1)
        self.conv2 = conv(self.ndf, self.ndf, kernel=5, pad=2, dil=1)
        self.conv3 = conv_downsample(self.ndf, self.ndf * 2)
        self.conv4 = conv(self.ndf * 2, self.ndf * 2, kernel=3, pad=1, dil=self.dil)
        self.conv5 = conv(self.ndf * 2, self.ndf * 2, kernel=3, pad=1, dil=self.dil)
        self.conv6 = conv_downsample(self.ndf * 2, self.ndf * 4)
        self.conv7 = conv(self.ndf * 4, self.ndf * 4, kernel=3, pad=1, dil=self.dil)
        self.conv8 = conv(self.ndf * 4, self.ndf * 4, kernel=3, pad=1, dil=self.dil)
        self.conv9 = conv_downsample(self.ndf * 4, self.ndf * 8)

    def forward(self, x):
        out = self.conv1(x);
        out_pre_ds_1 = self.conv2(out);
        out = self.conv3(out);
        out = self.conv4(out);
        out_pre_ds_2 = self.conv5(out);
        out = self.conv6(out);
        out = self.conv7(out);
        out_pre_ds_3 = self.conv8(out);
        out = self.conv9(out);

        return out_pre_ds_1, out_pre_ds_2, out_pre_ds_3, out


class Decoder(nn.Module):
    def __init__(self, ndf, dim):
        super(Decoder, self).__init__()

        self.dim = dim
        self.ndf = ndf

        self.upsample3 = Interpolate(size=(dim // 4, dim // 4), mode='bilinear')
        self.tconv8 = conv(self.ndf * 8, self.ndf * 4, kernel=3, pad=1, dil=1)
        self.conv1x1_3 = conv_1x1(self.ndf * 8, self.ndf * 4)
        self.tconv7 = conv(self.ndf * 4, self.ndf * 4, kernel=3, pad=1, dil=1)
        self.tconv6 = conv(self.ndf * 4, self.ndf * 4, kernel=3, pad=1, dil=1)
        self.upsample2 = Interpolate(size=(dim // 2, dim // 2), mode='bilinear')
        self.tconv5 = conv(self.ndf * 4, self.ndf * 2, kernel=3, pad=1, dil=1)
        self.conv1x1_2 = conv_1x1(self.ndf * 4, self.ndf * 2)
        self.tconv4 = conv(self.ndf * 2, self.ndf * 2, kernel=3, pad=1, dil=1)
        self.tconv3 = conv(self.ndf * 2, self.ndf * 2, kernel=3, pad=1, dil=1)
        self.upsample1 = Interpolate(size=(dim, dim), mode='bilinear')
        self.tconv2 = conv(self.ndf * 2, self.ndf, kernel=3, pad=1, dil=1)
        self.conv1x1_1 = conv_1x1(self.ndf * 2, self.ndf)
        self.tconv1 = conv_seg(self.ndf, 9)


    def forward(self, x):
        out = self.upsample3(x[3])
        out = self.tconv8(out)
        out_cat3 = torch.cat((out, x[2]), 1)
        out = self.conv1x1_3(out_cat3)
        out = self.tconv7(out)
        out = self.tconv6(out)
        out = self.upsample2(out)
        out = self.tconv5(out)
        out_cat2 = torch.cat((out, x[1]), 1)
        out = self.conv1x1_2(out_cat2)
        out = self.tconv4(out)
        out = self.tconv3(out)
        out = self.upsample1(out)
        out = self.tconv2(out)
        out_cat1 = torch.cat((out, x[0]), 1)
        out = self.conv1x1_1(out_cat1)
        out = self.tconv1(out)

        return out


class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size, bias=True):
        super(ConvLSTMCell, self).__init__()

        assert hidden_channels % 2 == 0

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.bias = bias
        self.kernel_size = kernel_size
        self.num_features = 4

        self.padding = int((kernel_size - 1) / 2)

        self.Wxi = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whi = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxf = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whf = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxc = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whc = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxo = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding,  bias=True)
        self.Who = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)

        self.Wci = None
        self.Wcf = None
        self.Wco = None

    def forward(self, x, h, c):
        ci = torch.sigmoid(self.Wxi(x) + self.Whi(h) + c * self.Wci)
        cf = torch.sigmoid(self.Wxf(x) + self.Whf(h) + c * self.Wcf)
        cc = cf * c + ci * torch.tanh(self.Wxc(x) + self.Whc(h))
        co = torch.sigmoid(self.Wxo(x) + self.Who(h) + cc * self.Wco)
        ch = co * torch.tanh(cc)
        return ch, cc

    def init_hidden(self, batch_size, hidden, shape):
        self.Wci = Variable(torch.zeros(1, hidden, shape[0], shape[1])).cuda()
        self.Wcf = Variable(torch.zeros(1, hidden, shape[0], shape[1])).cuda()
        self.Wco = Variable(torch.zeros(1, hidden, shape[0], shape[1])).cuda()
        return (Variable(torch.zeros(batch_size, hidden, shape[0], shape[1])).cuda(),
                Variable(torch.zeros(batch_size, hidden, shape[0], shape[1])).cuda())


class ConvLSTM(nn.Module):
    # input_channels corresponds to the first input feature map
    # hidden state is a list of succeeding lstm layers.
    def __init__(self, input_channels, hidden_channels, kernel_size, step=1, effective_step=[1], bias=True):
        super(ConvLSTM, self).__init__()
        self.input_channels = [input_channels] + hidden_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = len(hidden_channels)
        self.step = step
        self.bias = bias
        self.effective_step = effective_step
        self._all_layers = []
        for i in range(self.num_layers):
            name = 'cell{}'.format(i)
            cell = ConvLSTMCell(self.input_channels[i], self.hidden_channels[i], self.kernel_size, self.bias)
            setattr(self, name, cell)
            self._all_layers.append(cell)

    def forward(self, input):
        internal_state = []
        outputs = []
        for step in range(self.step):
            x = input
            for i in range(self.num_layers):
                # all cells are initialized in the first step
                name = 'cell{}'.format(i)
                if step == 0:
                    bsize, _, height, width = x.size()
                    (h, c) = getattr(self, name).init_hidden(batch_size=bsize, hidden=self.hidden_channels[i], shape=(height, width))
                    internal_state.append((h, c))

                # do forward
                (h, c) = internal_state[i]
                x, new_c = getattr(self, name)(x, h, c)
                internal_state[i] = (x, new_c)
            # only record effective steps
            if step in self.effective_step:
                outputs.append(x)

        return outputs, (x, new_c)


class Seq2Seg(nn.Module):
    def __init__(self, dim, ndf, dil):
        super(Seq2Seg, self).__init__()

        self.dim = dim
        self.ndf = ndf
        self.dil = dil

        self.encoder = Encoder(self.ndf, self.dim)
        self.convlstm =  ConvLSTM(input_channels=64, hidden_channels=[64, 64], kernel_size=3, step=5, effective_step=[4])
        self.decoder = Decoder(self.ndf, self.dim)

    def forward(self, x):
        out_list = []
        out_pre_ds_1, out_pre_ds_2, out_pre_ds_3, out = self.encoder(x)
       
        out, _ = self.convlstm(out)

        out_list.append(out_pre_ds_1)
        out_list.append(out_pre_ds_2)
        out_list.append(out_pre_ds_3)
        out_list.append(out[0])

        out = self.decoder(out_list)

        return out

