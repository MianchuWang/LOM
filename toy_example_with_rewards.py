import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# Check for CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Generate double circle data with rewards
def generate_double_circle_data(n_samples=5000, inner_radius=0.3, outer_radius=1.0, width=0.1):
    data = []
    rewards = []
    n_inner = n_samples // 4
    n_outer = n_samples - n_inner

    # Inner circle
    angles_inner = np.random.rand(n_inner) * 2 * np.pi
    radii_inner = inner_radius + width * np.random.rand(n_inner)
    x_inner = radii_inner * np.cos(angles_inner)
    y_inner = radii_inner * np.sin(angles_inner)
    for angle in angles_inner:
        if 0 <= angle < np.pi/2:
            rewards.append(1)
        elif np.pi/2 <= angle < np.pi:
            rewards.append(0)
        elif np.pi <= angle < 3*np.pi/2:
            rewards.append(1)
        else:
            rewards.append(0)
    data.extend(np.stack((x_inner, y_inner), axis=1))

    # Outer circle
    angles_outer = np.random.rand(n_outer) * 2 * np.pi
    radii_outer = outer_radius + width * np.random.rand(n_outer)
    x_outer = radii_outer * np.cos(angles_outer)
    y_outer = radii_outer * np.sin(angles_outer)
    for angle in angles_outer:
        if 0 <= angle < np.pi/2:
            rewards.append(1)
        elif np.pi/2 <= angle < np.pi:
            rewards.append(0)
        elif np.pi <= angle < 3*np.pi/2:
            rewards.append(1)
        else:
            rewards.append(0)
    data.extend(np.stack((x_outer, y_outer), axis=1))

    return np.array(data), np.array(rewards)

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

def gaussian_loss(mean, log_std, y, r):
    std = torch.exp(log_std)
    return torch.mean(torch.exp(2 * r) * 0.5 * ((y - mean) / std) ** 2 + log_std)

def train_gaussian_network(data_x, data_y, data_r, n_epochs=1000):
    model = GaussianNetwork(input_dim=data_x.shape[1], output_dim=data_y.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        mean, log_std = model(data_x)
        loss = gaussian_loss(mean, log_std, data_y, data_r)
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Gaussian Network Loss: {loss.item()}')
    
    return model

def sample_from_gaussian_network(model, x, n_samples=100):
    mean, log_std = model(x)
    std = torch.exp(log_std).detach().cpu().numpy()
    mean = mean.detach().cpu().numpy()
    
    samples = []
    for _ in range(n_samples):
        for i in range(x.shape[0]):
            sample = np.random.normal(mean[i], std[i])
            samples.append([x[i].item(), sample.item()])
    return np.array(samples)

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

def cvae_loss(recon_y, y, mu, logvar, data_r):
    BCE = nn.functional.mse_loss(recon_y, y, reduction='sum')
    #BCE = torch.exp(data_r) * (recon_y - y).pow(2).mean(dim=1)
    BCE = (torch.exp(5*data_r) * (recon_y - y).pow(2)).sum()
    KLD = - torch.sum(torch.exp(5*data_r) * 0.5 * (1 + logvar - mu.pow(2) - logvar.exp()))
    return BCE + 0.1 * KLD

def train_cvae(data_x, data_y, data_r, latent_dim, n_epochs=20000):
    cvae = CVAE(input_dim=1, latent_dim=latent_dim, output_dim=1).to('cuda')
    optimizer = optim.Adam(cvae.parameters(), lr=0.001)
    
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        recon_y, mu, logvar = cvae(data_x, data_y)
        loss = cvae_loss(recon_y, data_y, mu, logvar, data_r)
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print(f'Epoch {epoch}, CVAE Loss: {loss.item()}')
    
    return cvae

# Sample from CVAE
def sample_from_cvae(model, x, n_samples=100):
    samples = []
    for _ in range(n_samples):
        z = torch.randn(x.shape[0], 10).to(device='cuda')
        y = model.decode(z, x).detach().cpu().numpy()
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

def train_cgan(data_x, data_y, data_r, n_epochs=20000):
    generator = Generator(input_dim=1, output_dim=1).to(device='cuda')
    discriminator = Discriminator(input_dim=1).to(device='cuda')
    
    optimizer_g = optim.Adam(generator.parameters(), lr=0.001)
    optimizer_d = optim.Adam(discriminator.parameters(), lr=0.001)
    
    for epoch in range(n_epochs):
        # Train Discriminator
        optimizer_d.zero_grad()
        outputs_real = discriminator(data_x, data_y)
        
        z = torch.randn(data_x.size(0), 1).to(device='cuda')
        fake_y = generator(data_x, z)
        outputs_fake = discriminator(data_x, fake_y)
        
        d_loss = - torch.exp(2.0 * data_r) * torch.log(outputs_real) - torch.log(1-outputs_fake)
        d_loss = d_loss.mean()
        d_loss.backward()
        optimizer_d.step()
        
        # Train Generator
        optimizer_g.zero_grad()
        z = torch.randn(data_x.size(0), 1).to(device='cuda')
        fake_y = generator(data_x, z)
        outputs = discriminator(data_x, fake_y)
        
        g_loss = torch.log(1 - outputs).mean()
        g_loss.backward()
        optimizer_g.step()
        
        if epoch % 100 == 0:
            print(f'Epoch {epoch}, D Loss: {d_loss.item()}, G Loss: {g_loss.item()}')
    
    return generator

# Sample from CGAN
def sample_from_cgan(generator, x, n_samples=100):
    samples = []
    for _ in range(n_samples):
        z = torch.randn(x.shape[0], 1).to(device='cuda')
        y = generator(x, z).detach().cpu().numpy()
        for i in range(x.shape[0]):
            samples.append([x[i].item(), y[i, 0]])
    return np.array(samples)

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
    alpha = alpha.detach().cpu().numpy()
    mu = mu.detach().cpu().numpy()
    sigma = sigma.detach().cpu().numpy()
    
    samples = []
    for _ in range(n_samples):
        for i in range(x.shape[0]):
            idx = np.random.choice(np.arange(alpha.shape[1]), p=alpha[i])
            sample = np.random.normal(mu[i, idx], sigma[i, idx])
            samples.append([x[i].item(), sample])
    return np.array(samples)
'''

def sample_from_mdn(model, x, n_samples=100):
    alpha, mu, sigma = model(x)
    alpha = alpha.detach().cpu().numpy()
    mu = mu.detach().cpu().numpy()
    sigma = sigma.detach().cpu().numpy()

    samples = []
    for _ in range(n_samples):
        for i in range(x.shape[0]):
            # Determine which components have high reward
            high_reward_indices = []
            for j in range(mu.shape[1]):
                angle = np.arctan2(mu[i, j], x[i].item())
                if (0 <= angle < np.pi/2) or (-np.pi <= angle < -np.pi/2):
                    high_reward_indices.append(j)
                        
            if not high_reward_indices:
                continue
            
            # Normalize the alpha values to sum to 1 over the high reward indices
            high_reward_alpha = alpha[i][high_reward_indices]
            high_reward_alpha /= np.sum(high_reward_alpha)
            
            # Sample from the high reward components
            idx = np.random.choice(high_reward_indices, p=high_reward_alpha)
            sample = np.random.normal(mu[i, idx], sigma[i, idx])
            samples.append([x[i].item(), sample])
    return np.array(samples)
'''



def train_mdn(data_x, data_y, n_mixtures, n_epochs=1000):
    mdn = MDN(input_dim=1, output_dim=1, n_mixtures=n_mixtures).to(device='cuda')
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

# Generate and scale the data
data, rewards = generate_double_circle_data(width=0.2)  # Adjust the width parameter as needed

# Convert data to PyTorch tensors
data_x = torch.tensor(data[:, 0:1], dtype=torch.float32).to(device)
data_y = torch.tensor(data[:, 1:], dtype=torch.float32).to(device)
data_r = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(device)

data_r = torch.zeros_like(data_r)
'''
# Train the Gaussian network
model = train_gaussian_network(data_x, data_y, data_r, n_epochs=10000) # 10000
# Sample actions from the trained policy
sampled_actions = sample_from_gaussian_network(model, data_x, n_samples=10)

cvae = train_cvae(data_x, data_y, data_r, latent_dim=10, n_epochs=50000) #50000
samples_cvae = sample_from_cvae(cvae, data_x, n_samples=10)

generator = train_cgan(data_x, data_y, data_r, n_epochs=50000) # 20000
samples_cgan = sample_from_cgan(generator, data_x, n_samples=10)

mdn = train_mdn(data_x, data_y, n_mixtures=20, n_epochs=10000) # 10000
samples_mdn = sample_from_mdn(mdn, data_x, n_samples=20)
'''
cmap = 'Blues'

# Plot the original data
plt.figure(figsize=(8, 8))
colors = ['blue', 'blue']
for reward in np.unique(rewards):
    indices = rewards == reward
    plt.scatter(data[indices, 0], data[indices, 1], s=1, label=f'Reward {reward}', color=colors[reward])
plt.title("Double Circle Data without Rewards", fontsize=20)
plt.xlabel("State", fontsize=16)
plt.ylabel("Action", fontsize=16)
plt.xlim(-1.6, 1.6)
plt.ylim(-1.6, 1.6)
#plt.legend(fontsize=14)
plt.grid(True)
plt.show()

# KDE plot of the sampled actions
plt.figure(figsize=(8, 8))
sns.kdeplot(x=sampled_actions[:, 0], y=sampled_actions[:, 1], cmap=cmap, 
            fill=True, bw_adjust=0.5, thresh=0.2)
plt.title("Sampled Actions from Gaussian", fontsize=20)
plt.xlabel("State", fontsize=16)
plt.ylabel("Action", fontsize=16)
plt.xlim(-1.6, 1.6)
plt.ylim(-1.6, 1.6)
plt.grid(True)
plt.show()

# KDE plot of the sampled actions
plt.figure(figsize=(8, 8))
sns.kdeplot(x=samples_cvae[:, 0], y=samples_cvae[:, 1], cmap=cmap, 
            fill=True, bw_adjust=0.5, thresh=0.2)
plt.title("Sampled Actions from CVAE", fontsize=20)
plt.xlabel("State", fontsize=16)
plt.ylabel("Action", fontsize=16)
plt.xlim(-1.6, 1.6)
plt.ylim(-1.6, 1.6)
plt.grid(True)
plt.show()


# KDE plot of the sampled actions
plt.figure(figsize=(8, 8))
sns.kdeplot(x=samples_cgan[:, 0], y=samples_cgan[:, 1], cmap=cmap, 
            fill=True, bw_adjust=0.5, thresh=0.2)
plt.title("Sampled Actions from CGAN", fontsize=20)
plt.xlabel("State", fontsize=16)
plt.ylabel("Action", fontsize=16)
plt.xlim(-1.6, 1.6)
plt.ylim(-1.6, 1.6)
plt.grid(True)
plt.show()



# KDE plot of the sampled actions
plt.figure(figsize=(8, 8))
sns.kdeplot(x=samples_mdn[:, 0], y=samples_mdn[:, 1], cmap=cmap, 
            fill=True, bw_adjust=0.5, thresh=0.2)
plt.title("Sampled Actions from MDN", fontsize=20)
plt.xlabel("State", fontsize=16)
plt.ylabel("Action", fontsize=16)
plt.xlim(-1.6, 1.6)
plt.ylim(-1.6, 1.6)
plt.grid(True)
plt.show()


