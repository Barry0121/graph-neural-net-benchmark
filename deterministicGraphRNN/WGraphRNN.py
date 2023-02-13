import torch
import torch.optim as optim

# Outline:
#
# 1.

# some argument parser thing, should be replaced
opt = {}
opt.epochs = 0
opt.batch_size = 0
opt.clamp_lower, opt.clamp_upper = -0.01, 0.01 # weight clippers
opt.lrD = 0 # learning rate for netD
opt.lrG = 0
opt.betasD = () # parameters for optimizerD
opt.betasG = () # parameters for optimizerG
opt.nz = 0 # size of noise vector

dataloader = None # some torch.utils.data.DataLoader object

# things to call backward on for netD
one = torch.FloatTensor([1])
mone = one * -1
noise = None # change its dimensions to those of outputs from inverter

# declare discriminator, generator
netD = None # discriminator network that plays the role of a 1-Lipschitz function, is a NN
netG = None # GraphRNN

# declare optimizers
optimizerD = optim.Adam(netD.parameters(), lr=opt.lrD, betas=opt.betasD)
optimizerG = optim.Adam(netG.parameters(), lr=opt.lrG, betas=opt.betasG)

# pulled from 156-212 in main.py of WassersteinGAN
gen_iterations = 0 # iterations of updates for netG
for epoch in range(opt.epochs):
    data_iter = iter(dataloader)
    i = 0
    while i < len(dataloader): # why did WGAN authors not just... iterate through the dataloader?
        ########################
        # (1) Update D network #
        ########################
        for p in netD.parameters(): # reset requires_grad
            p.requires_grad = True # they are set to False below in netG update
        Diters = 0 # number of iterations to train discriminator
        j = 0 # counter for 1, 2, ... Diters
        while j < Diters and i < len(dataloader):
            j += 1

            # weight clipping: clamp parameters to a cube
            for p in netD.parameters():
                p.data.clamp_(opt.clamp_lower, opt.clamp_upper)

            data = data_iter.next()
            i += 1
            netD.zero_grad()

            # train with real
            input = data.copy()
                # insert data processing
            errD_real = netD(input)
            errD_real.backward(one) # discriminator should assign 1's to true samples

            # train with fake
            input = noise.resize_(opt.batch_size, 1).normal_(0, 1)
                # insert data processing
            fake = netG(input)
            errD_fake = netD(fake)
            errD_fake.backward(mone) # discriminator should assign -1's to fake samples??

            # compute Wasserstein distance and update parameters
            errD = errD_real - errD_fake
            optimizerD.step()

        ########################
        # (2) Update G network #
        ########################

        for p in netD.parameters():
            p.requires_grad = False # to avoid computation
        netG.zero_grad()

        input = noise.resize_(opt.batch_size, 1).normal_(0, 1)
            # insert data processing
        fake = netG(input)
        errG = netD(fake) # <- critic's opinion; output of solution f as in WGAN Theorem 3

        # update netG parameters
        errG.backward(one)
        optimizerG.step()

        gen_iterations += 1

        # maybe insert some statistics to print during training