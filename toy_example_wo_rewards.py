import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim

def generate_double_circle_data(n_samples=5000, inner_radius=0.3, outer_radius=1.0):
    data = []
    n_inner = n_samples // 4
    n_outer = n_samples - n_inner

    # Inner circle
    angles_inner = np.random.rand(n_inner) * 2 * np.pi
    x_inner = inner_radius * np.cos(angles_inner)
    y_inner = inner_radius * np.sin(angles_inner)
    data.extend(np.stack((x_inner, y_inner), axis=1))

    # Outer circle
    angles_outer = np.random.rand(n_outer) * 2 * np.pi
    x_outer = outer_radius * np.cos(angles_outer)
    y_outer = outer_radius * np.sin(angles_outer)
    data.extend(np.stack((x_outer, y_outer), axis=1))

    return np.array(data)

class MDN(nn.Module):
    def __init__(self, input_dim, output_dim, n_mixtures):
        super(MDN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.alpha = nn.Linear(64, n_mixtures)
        self.mu = nn.Linear(64, n_mixtures * output_dim)
        self.sigma = nn.Linear(64, n_mixtures * output_dim)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        alpha = torch.softmax(self.alpha(x), dim=1)
        mu = self.mu(x)
        sigma = torch.exp(self.sigma(x))
        return alpha, mu, sigma

def mdn_loss(alpha, mu, sigma, y):
    m = alpha * torch.exp(-0.5 * ((y - mu) / sigma) ** 2) / sigma
    loss = -torch.log(torch.sum(m, dim=1))
    return torch.mean(loss)

def sample_from_mdn(model, x, n_samples=100):
    alpha, mu, sigma = model(x)
    alpha = alpha.detach().numpy()
    mu = mu.detach().numpy()
    sigma = sigma.detach().numpy()
    
    samples = []
    for _ in range(n_samples):
        for i in range(x.shape[0]):
            idx = np.random.choice(np.arange(alpha.shape[1]), p=alpha[i])
            sample = np.random.normal(mu[i, idx], sigma[i, idx])
            samples.append([x[i].item(), sample])
    return np.array(samples)

def train_mdn(data_x, data_y, n_mixtures, n_epochs=1000):
    mdn = MDN(input_dim=1, output_dim=1, n_mixtures=n_mixtures)
    optimizer = optim.Adam(mdn.parameters(), lr=0.001)
    
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        alpha, mu, sigma = mdn(data_x)
        loss = mdn_loss(alpha, mu, sigma, data_y)
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Mixtures {n_mixtures}, Loss: {loss.item()}')
    
    return mdn


class CVAE(nn.Module):
    def __init__(self, input_dim, latent_dim, output_dim):
        super(CVAE, self).__init__()
        self.fc1 = nn.Linear(input_dim + output_dim, 64)
        self.fc21 = nn.Linear(64, latent_dim)  # mu
        self.fc22 = nn.Linear(64, latent_dim)  # logvar
        self.fc3 = nn.Linear(latent_dim + input_dim, 64)
        self.fc4 = nn.Linear(64, output_dim)
    
    def encode(self, x):
        h1 = torch.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z, y):
        h3 = torch.relu(self.fc3(torch.cat([z, y], 1)))
        return self.fc4(h3)
    
    def forward(self, x, y):
        mu, logvar = self.encode(torch.cat([x, y], 1))
        z = self.reparameterize(mu, logvar)
        return self.decode(z, x), mu, logvar

def cvae_loss(recon_y, y, mu, logvar):
    BCE = nn.functional.mse_loss(recon_y, y, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + 0.1 * KLD

def train_cvae(data_x, data_y, latent_dim, n_epochs=20000):
    cvae = CVAE(input_dim=1, latent_dim=latent_dim, output_dim=1)
    optimizer = optim.Adam(cvae.parameters(), lr=0.001)
    
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        recon_y, mu, logvar = cvae(data_x, data_y)
        loss = cvae_loss(recon_y, data_y, mu, logvar)
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print(f'Epoch {epoch}, CVAE Loss: {loss.item()}')
    
    return cvae

# Sample from CVAE
def sample_from_cvae(model, x, n_samples=100):
    samples = []
    for _ in range(n_samples):
        z = torch.randn(x.shape[0], 2)
        y = model.decode(z, x).detach().numpy()
        for i in range(x.shape[0]):
            samples.append([x[i].item(), y[i, 0]])
    return np.array(samples)


class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(input_dim + 1, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)
    
    def forward(self, x, y):
        x = torch.cat([x, y], 1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(input_dim + 1, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)
    
    def forward(self, x, y):
        x = torch.cat([x, y], 1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.sigmoid(self.fc3(x))

def train_cgan(data_x, data_y, n_epochs=20000):
    generator = Generator(input_dim=1, output_dim=1)
    discriminator = Discriminator(input_dim=1)
    
    criterion = nn.BCELoss()
    optimizer_g = optim.Adam(generator.parameters(), lr=0.001)
    optimizer_d = optim.Adam(discriminator.parameters(), lr=0.001)
    
    for epoch in range(n_epochs):
        # Train Discriminator
        optimizer_d.zero_grad()
        real_labels = torch.ones(data_x.size(0), 1)
        fake_labels = torch.zeros(data_x.size(0), 1)
        
        outputs = discriminator(data_x, data_y)
        d_loss_real = criterion(outputs, real_labels)
        
        z = torch.randn(data_x.size(0), 1)
        fake_y = generator(data_x, z)
        outputs = discriminator(data_x, fake_y)
        d_loss_fake = criterion(outputs, fake_labels)
        
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        optimizer_d.step()
        
        # Train Generator
        optimizer_g.zero_grad()
        z = torch.randn(data_x.size(0), 1)
        fake_y = generator(data_x, z)
        outputs = discriminator(data_x, fake_y)
        
        g_loss = criterion(outputs, real_labels)
        g_loss.backward()
        optimizer_g.step()
        
        if epoch % 100 == 0:
            print(f'Epoch {epoch}, D Loss: {d_loss.item()}, G Loss: {g_loss.item()}')
    
    return generator

# Sample from CGAN
def sample_from_cgan(generator, x, n_samples=100):
    samples = []
    for _ in range(n_samples):
        z = torch.randn(x.shape[0], 1)
        y = generator(x, z).detach().numpy()
        for i in range(x.shape[0]):
            samples.append([x[i].item(), y[i, 0]])
    return np.array(samples)


class GaussianNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GaussianNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.mean = nn.Linear(64, output_dim)
        self.log_std = nn.Linear(64, output_dim)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = self.log_std(x)
        return mean, log_std

def gaussian_loss(mean, log_std, y):
    std = torch.exp(log_std)
    return torch.mean(0.5 * ((y - mean) / std) ** 2 + log_std)

def train_gaussian_network(data_x, data_y, n_epochs=1000):
    model = GaussianNetwork(input_dim=1, output_dim=1)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        mean, log_std = model(data_x)
        loss = gaussian_loss(mean, log_std, data_y)
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Gaussian Network Loss: {loss.item()}')
    
    return model

def sample_from_gaussian_network(model, x, n_samples=100):
    mean, log_std = model(x)
    std = torch.exp(log_std).detach().numpy()
    mean = mean.detach().numpy()
    
    samples = []
    for _ in range(n_samples):
        for i in range(x.shape[0]):
            sample = np.random.normal(mean[i], std[i])
            samples.append([x[i].item(), sample.item()])
    return np.array(samples)

data = generate_double_circle_data()
data_x = torch.tensor(data[:, 0], dtype=torch.float32).reshape(-1, 1)
data_y = torch.tensor(data[:, 1], dtype=torch.float32).reshape(-1, 1)
x_values = torch.linspace(-1.1, 1.1, 200).reshape(-1, 1)


generator = train_cgan(data_x, data_y, n_epochs=20000)
samples_cgan = sample_from_cgan(generator, x_values, n_samples=50)

gaussian_net = train_gaussian_network(data_x, data_y, n_epochs=1000)
samples_gaussian = sample_from_gaussian_network(gaussian_net, x_values, n_samples=50)

cvae = train_cvae(data_x, data_y, latent_dim=2, n_epochs=10000)
samples_cvae = sample_from_cvae(cvae, x_values, n_samples=50)

mdn_1 = train_mdn(data_x, data_y, n_mixtures=1, n_epochs=1000)
samples_1 = sample_from_mdn(mdn_1, x_values, n_samples=50)

mdn_4 = train_mdn(data_x, data_y, n_mixtures=4, n_epochs=1000)
samples_4 = sample_from_mdn(mdn_4, x_values, n_samples=50)

mdn_10 = train_mdn(data_x, data_y, n_mixtures=10, n_epochs=1000)
samples_10 = sample_from_mdn(mdn_10, x_values, n_samples=50)

mdn_20 = train_mdn(data_x, data_y, n_mixtures=20, n_epochs=1000)
samples_20 = sample_from_mdn(mdn_20, x_values, n_samples=50)


plt.figure(figsize=(40, 16))

# KDE of Double Circle Data
plt.subplot(2, 4, 1)
sns.kdeplot(x=data[:, 0], y=data[:, 1], fill=True, cmap="Blues", thresh=0.1, levels=100)
plt.title("KDE of Double Circle Data", fontsize=20)
plt.xlabel("State", fontsize=16)
plt.ylabel("Action", fontsize=16)
plt.xlim(-1.6, 1.6)
plt.ylim(-1.6, 1.6)
plt.grid(True, zorder=0)

# Gaussian Model
plt.subplot(2, 4, 2)
sns.kdeplot(x=samples_gaussian[:, 0], y=samples_gaussian[:, 1], fill=True, cmap="Blues", thresh=0.1, levels=100)
plt.title("Gaussian Model", fontsize=20)
plt.xlabel("State", fontsize=16)
plt.ylabel("Action", fontsize=16)
plt.xlim(-1.6, 1.6)
plt.ylim(-1.6, 1.6)
plt.grid(True, zorder=0)

# CVAE
plt.subplot(2, 4, 3)
sns.kdeplot(x=samples_cvae[:, 0], y=samples_cvae[:, 1], fill=True, cmap="Blues", thresh=0.1, levels=100)
plt.title("CVAE", fontsize=20)
plt.xlabel("State", fontsize=16)
plt.ylabel("Action", fontsize=16)
plt.xlim(-1.6, 1.6)
plt.ylim(-1.6, 1.6)
plt.grid(True, zorder=0)

# CGAN
plt.subplot(2, 4, 4)
sns.kdeplot(x=samples_cgan[:, 0], y=samples_cgan[:, 1], fill=True, cmap="Blues", thresh=0.1, levels=100)
plt.title("CGAN", fontsize=20)
plt.xlabel("State", fontsize=16)
plt.ylabel("Action", fontsize=16)
plt.xlim(-1.6, 1.6)
plt.ylim(-1.6, 1.6)
plt.grid(True, zorder=0)

# MDN with 1 mixture
plt.subplot(2, 4, 5)
sns.kdeplot(x=samples_1[:, 0], y=samples_1[:, 1], fill=True, cmap="Blues", thresh=0.1, levels=100)
plt.title("MDN with 1 Mixture", fontsize=20)
plt.xlabel("State", fontsize=16)
plt.ylabel("Action", fontsize=16)
plt.xlim(-1.6, 1.6)
plt.ylim(-1.6, 1.6)
plt.grid(True, zorder=0)

# MDN with 4 mixtures
plt.subplot(2, 4, 6)
sns.kdeplot(x=samples_4[:, 0], y=samples_4[:, 1], fill=True, cmap="Blues", thresh=0.1, levels=100)
plt.title("MDN with 4 Mixtures", fontsize=20)
plt.xlabel("State", fontsize=16)
plt.ylabel("Action", fontsize=16)
plt.xlim(-1.6, 1.6)
plt.ylim(-1.6, 1.6)
plt.grid(True, zorder=0)

# MDN with 10 mixtures
plt.subplot(2, 4, 7)
sns.kdeplot(x=samples_10[:, 0], y=samples_10[:, 1], fill=True, cmap="Blues", thresh=0.1, levels=100)
plt.title("MDN with 10 Mixtures", fontsize=20)
plt.xlabel("State", fontsize=16)
plt.ylabel("Action", fontsize=16)
plt.xlim(-1.6, 1.6)
plt.ylim(-1.6, 1.6)
plt.grid(True, zorder=0)

# MDN with 20 mixtures
plt.subplot(2, 4, 8)
sns.kdeplot(x=samples_20[:, 0], y=samples_20[:, 1], fill=True, cmap="Blues", thresh=0.1, levels=100)
plt.title("MDN with 20 Mixtures", fontsize=20)
plt.xlabel("State", fontsize=16)
plt.ylabel("Action", fontsize=16)
plt.xlim(-1.6, 1.6)
plt.ylim(-1.6, 1.6)
plt.grid(True, zorder=0)

plt.tight_layout()
plt.show()
