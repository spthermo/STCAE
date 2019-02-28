import torch
import torch.nn as nn
from model_utils import *

class Encoder(nn.Module):
    def __init__(self, ndf, dil):
        super(Encoder, self).__init__()

        self.ndf = ndf
        self.dil = dil

        self.conv1 = conv(3, self.ndf, kernel=7, pad=3, dil=1)
        self.conv2 = conv(self.ndf, self.ndf, kernel=5, pad=2, dil=1)
        self.pool1 = maxpool(kernel=2, stride=2)
        self.conv3 = conv(self.ndf, self.ndf * 2, kernel=3, pad=1, dil=self.dil)
        self.conv4 = conv(self.ndf * 2, self.ndf * 2, kernel=3, pad=1, dil=self.dil)
        self.conv5 = conv(self.ndf * 2, self.ndf * 2, kernel=3, pad=1, dil=self.dil)
        self.pool2 = maxpool(kernel=2, stride=2)
        self.conv6 = conv(self.ndf * 4, self.ndf * 4, kernel=3, pad=1, dil=self.dil)
        self.conv7 = conv(self.ndf * 4, self.ndf * 4, kernel=3, pad=1, dil=self.dil)
        self.conv8 = conv(self.ndf * 4, self.ndf * 4, kernel=3, pad=1, dil=self.dil)
        self.pool3 = maxpool(kernel=2, stride=2)
        self.conv9 = conv(self.ndf * 4, self.ndf * 8, kernel=3, pad=1, dil=self.dil)
        self.conv10 = conv(self.ndf * 8, self.ndf * 8, kernel=3, pad=1, dil=self.dil)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(x)
        out_pre_ds_1 = self.pool1(out)
        out = self.conv3(out_pre_ds_1)
        out = self.conv4(out)
        out = self.conv5(out)
        out_pre_ds_2 = self.pool2(out)
        out = self.conv6(out_pre_ds_2)
        out = self.conv7(out)
        out = self.conv8(out)
        out_pre_ds_3 = self.pool3(out)
        out = self.conv9(out_pre_ds_3)
        out = self.conv10(out)

        return out_pre_ds_1, out_pre_ds_2, out_pre_ds_3, out


class Decoder(nn.Module):
    def __init__(self, ndf, dim):
        super(Decoder, self).__init__()

        self.dim = dim
        self.ndf = ndf

        self.tconv10 = conv(self.ndf * 8, self.ndf * 4, kernel=3, pad=1, dil=1)
        self.tconv9 = conv(self.ndf * 8, self.ndf * 4, kernel=3, pad=1, dil=1)
        self.upsample3 = Interpolate(size=(dim // 4, dim // 4), mode='bilinear')
        self.tconv_up3 = conv(self.ndf * 8, self.ndf * 4, kernel=3, pad=1, dil=1)
        self.conv1x1_3 = conv_1x1(self.ndf * 8, self.ndf * 4)
        self.tconv8 = conv(self.ndf * 4, self.ndf * 4, kernel=3, pad=1, dil=1)
        self.tconv7 = conv(self.ndf * 4, self.ndf * 4, kernel=3, pad=1, dil=1)
        self.tconv6 = conv(self.ndf * 4, self.ndf * 4, kernel=3, pad=1, dil=1)
        self.upsample2 = Interpolate(size=(dim // 2, dim // 2), mode='bilinear')
        self.tconv_up2 = conv(self.ndf * 4, self.ndf * 2, kernel=3, pad=1, dil=1)
        self.conv1x1_2 = conv_1x1(self.ndf * 4, self.ndf * 2)
        self.tconv5 = conv(self.ndf * 2, self.ndf * 2, kernel=3, pad=1, dil=1)
        self.tconv4 = conv(self.ndf * 2, self.ndf * 2, kernel=3, pad=1, dil=1)
        self.tconv3 = conv(self.ndf * 2, self.ndf * 2, kernel=3, pad=1, dil=1)
        self.upsample1 = Interpolate(size=(dim, dim), mode='bilinear')
        self.tconv_up1 = conv(self.ndf * 2, self.ndf, kernel=3, pad=1, dil=1)
        self.conv1x1_1 = conv_1x1(self.ndf * 2, self.ndf)
        self.tconv2 = conv(self.ndf, self.ndf, kernel=3, pad=1, dil=1)
        self.tconv1 = conv_seg(self.ndf, 9)


    def forward(self, x):
        out = self.tconv10(x[3])
        out = self.tconv9(out)
        out = self.upsample3(out)
        out = self.tconv_up3(out)
        out_cat3 = torch.cat((out, x[2]), 1)
        out = self.conv1x1_3(out_cat3)
        out = self.tconv8(out)
        out = self.tconv7(out)
        out = self.tconv6(out)
        out = self.upsample2(out)
        out = self.tconv_up2(out)
        out_cat2 = torch.cat((out, x[1]), 1)
        out = self.conv1x1_2(out_cat2)
        out = self.tconv5(out)
        out = self.tconv4(out)
        out = self.tconv3(out)
        out = self.upsample1(out)
        out = self.tconv_up1(out)
        out_cat1 = torch.cat((out, x[0]), 1)
        out = self.conv1x1_1(out_cat1)
        out = self.tconv2(out)
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

