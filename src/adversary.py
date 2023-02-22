

def generate_adversary(self):
    # Load the trained models.
    self.load_model(self.resume_iters)

    # Choose search method.
    print('Search method:', self.search)

    # Choose the classifier to generate adversary examples against.
    print('Classifier:', self.classifier)
    if self.classifier == 'lenet':
        C = LeNet().to(self.device) # TODO: change the model to GCN
        cla_path = os.path.join(self.cla_dir, self.classifier, '{}_lenet.ckpt'.format(self.cla_iters))
        C.load_state_dict(torch.load(cla_path, map_location=lambda storage, loc: storage))

    # Generate adversary examples.
    for j, (images, labels) in enumerate(self.test_loader):
        for i in range(32):
            x = images[i].unsqueeze(0).to(self.device)
            y = labels[i].to(self.device)

            adversary = self.iterative_search(self.G, self.I, C, x, y,
                                                n_samples=self.n_samples, step=self.step)
            adversary_path = os.path.join(self.adversary_dir,
                                '{}_{}_{}.jpg'.format(self.classifier, j+1, i+1))
            self.save_adversary(adversary, adversary_path)
            print('Saved natural adversary example...'.format(adversary_path))

def save_adversary(self, adversary, filename):
    fig, ax = plt.subplots(1, 2, figsize=(7, 3))

    ax[0].imshow(adversary['x'],
                    interpolation='none', cmap=plt.get_cmap('gray'))
    ax[0].text(1, 5, str(adversary['y']), color='white', fontsize=20)
    ax[0].axis('off')

    ax[1].imshow(adversary['x_adv'],
                    interpolation='none', cmap=plt.get_cmap('gray'))
    ax[1].text(1, 5, str(adversary['y_adv']), color='white', fontsize=20)
    ax[1].axis('off')

    fig.savefig(filename)
    plt.close()

def iterative_search(self, G, I, C, x, y, y_target=None, z=None,
                        n_samples=5000, step=0.01, l=0., h=10., ord=2):
    """
    :param G: function of generator
    :param I: function of inverter
    :param C: function of classifier
    :param x: input instance
    :param y: label
    :param y_target: target label for adversary
    :param z: latent vector corresponding to x
    :param n_samples: number of samples in each search iteration
    :param step: delta r for search step size
    :param l: lower bound of search range
    :param h: upper bound of search range
    :param ord: indicating norm order
    :return: adversary for x against the classifier (d_adv is delta z between z and z_adv)
    """

    x_adv, y_adv, z_adv, d_adv = None, None, None, None
    h = l + step

    if z is None:
        z = I(x)

    while True:
        delta_z = np.random.randn(n_samples, z.size(1))
        d = np.random.rand(n_samples) * (h - l) + l        # random values between the search range [l, h)
        norm_p = np.linalg.norm(delta_z, ord=ord, axis=1)  # L2 norm of delta z along axis=1
        d_norm = np.divide(d, norm_p).reshape(-1, 1)       # rescale/normalize d
        delta_z = np.multiply(delta_z, d_norm)             # random noise vectors of norms within (r, r + delta r]
        delta_z = torch.from_numpy(delta_z).float().to(self.device)
        z_tilde = z + delta_z
        x_tilde = G(z_tilde)
        y_tilde = torch.argmax(C(x_tilde), dim=1)

        if y_target is None:
            indices_adv = np.where(y_tilde != y)[0]
        else:
            indices_adv = np.where(y_tilde == y_target)[0]

        # No candidate generated.
        if len(indices_adv) == 0:
            print('No candidate generated, increasing search range...')
            l = h
            h = l + step

        # Certain candidates generated.
        else:
            # Choose the data index with the least perturbation.
            idx_adv = indices_adv[np.argmin(d[indices_adv])]

            # For debugging.
            if y_target is None:
                assert (y_tilde[idx_adv] != y)

            else:
                assert (y_tilde[idx_adv] == y_target)

            # Save natural adversary example.
            if d_adv is None or d[idx_adv] < d_adv:
                x_adv = x_tilde[idx_adv]
                y_adv = y_tilde[idx_adv]
                z_adv = z_tilde[idx_adv]
                d_adv = d[idx_adv]

                if y_target is None:
                    print("Untarget y=%d y_adv=%d d_adv=%.4f l=%.4f h=%.4f" % (y, y_adv, d_adv, l, h))
                else:
                    print("Targeted y=%d y_adv=%d d_adv=%.4f l=%.4f h=%.4f" % (y, y_adv, d_adv, l, h))

                break

    adversary = {'x': x.squeeze(0).squeeze(0).data.cpu().numpy(),
                    'y': y.data.cpu().numpy(),
                    'z': z.squeeze(0).data.cpu().numpy(),
                    'x_adv': x_adv.squeeze(0).data.cpu().numpy(),
                    'y_adv': y_adv.data.cpu().numpy(),
                    'z_adv': z_adv.data.cpu().numpy(),
                    'd_adv': d_adv}

    return adversary