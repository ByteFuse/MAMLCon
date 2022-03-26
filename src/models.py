import torch
import torch.nn as nn

class ProjectionHead(nn.Module):
    def __init__(
        self,
        embedding_dim,
        projection_dim
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.lrelu = nn.LeakyReLU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.layer_norm = nn.LayerNorm(projection_dim)
    
    def forward(self, x):
        projected = self.projection(x)
        x = self.lrelu(projected)
        x = self.fc(x)
        x = x + projected
        x = self.layer_norm(x)
        return x

class WordClassificationAudioCnnPool(nn.Module):
    def __init__(self, embedding_dim=128, hidden_dim=512, input_channels=20):
        super().__init__()

        conv1 = nn.Sequential(
            nn.Conv1d(in_channels=input_channels, out_channels=hidden_dim, kernel_size=10, stride=5),
            nn.BatchNorm1d(hidden_dim, track_running_stats=False),
            nn.ReLU(),
        )

        conv2 = nn.Sequential(
            nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=8, stride=4),
            nn.BatchNorm1d(hidden_dim, track_running_stats=False),
            nn.ReLU(),
        )
        conv3 = nn.Sequential(
            nn.Conv1d(in_channels=hidden_dim, out_channels=embedding_dim, kernel_size=4, stride=1),
        )

        self.encoder = nn.ModuleList([
            conv1, conv2, conv3
        ])  
        self.pool = torch.nn.AdaptiveMaxPool2d((embedding_dim,1))
        self.projection = ProjectionHead(embedding_dim, embedding_dim)

    def forward(self, audio):
        for conv in self.encoder:
            audio = conv(audio)

        embedding = self.pool(audio).squeeze(-1)
        embedding = self.projection(embedding)
        return embedding


class WordClassificationAudio2DCnn(nn.Module):
    def __init__(self, embedding_dim=128, hidden_dim=64, input_channels=32):
        super().__init__()

        conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=hidden_dim, kernel_size=(3,3), stride=(2,2), padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_dim, track_running_stats=False),
        )

        conv2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=(3,3), stride=(2,2), padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_dim, track_running_stats=False),
        )

        conv3 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=(3,3), stride=(2,2), padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_dim, track_running_stats=False),
        )

        conv4 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=(3,3), stride=(2,2), padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_dim, track_running_stats=False),
        )

        self.encoder = nn.ModuleList([
            conv1, conv2, conv3, conv4
        ])  

        self.flatten = nn.Flatten()

        if input_channels == 39:
          self.projection = nn.Linear(1344, embedding_dim)
        else:
          self.projection = nn.Linear(3584, embedding_dim)

    def forward(self, audio):
        # adding in channel batch if not there
        if len(audio.shape)!=4:
            audio = audio.unsqueeze(1)

        for conv in self.encoder:
            audio = conv(audio)
        
        audio = self.flatten(audio)
        embedding = self.projection(audio)

        return embedding


class WordClassificationAudioCnn(nn.Module):
    def __init__(self, embedding_dim=128, hidden_dim=512, input_channels=128):
        super().__init__()

        conv1 = nn.Sequential(
            nn.Conv1d(in_channels=input_channels, out_channels=hidden_dim, kernel_size=10, stride=5),
            nn.BatchNorm1d(hidden_dim, track_running_stats=False),
            nn.ReLU()
        )

        conv2 = nn.Sequential(
            nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=8, stride=4),
            nn.BatchNorm1d(hidden_dim, track_running_stats=False),
            nn.ReLU()
        )
        conv3 = nn.Sequential(
            nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=4, stride=1),
            nn.BatchNorm1d(hidden_dim, track_running_stats=False),
            nn.ReLU()
        )

        self.encoder = nn.ModuleList([
            conv1, conv2, conv3
        ])  
        
        self.flatten = nn.Flatten()
        self.projection = nn.Linear(1536, embedding_dim)

    def forward(self, audio):
        for conv in self.encoder:
            audio = conv(audio)
        audio = self.flatten(audio)
        embedding = self.projection(audio)
        return embedding


class WordClassificationRnn(nn.Module):
    def __init__(self, embedding_dim=128, hidden_dim=512, input_size=128, n_layers=3, learn_states=True):
        super().__init__()

        self.learn_states = learn_states
        self.hidden_dim = hidden_dim

        self.encoder =  nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=0,
            bidirectional=True
        )

        if self.learn_states:    
            self.ho = nn.Parameter(torch.randn(2*n_layers, 1, hidden_dim))
            self.co = nn.Parameter(torch.randn(2*n_layers, 1, hidden_dim))

        
        self.projection_head = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim*2, embedding_dim)
        )

    def forward(self, audio):
        batch_size = audio.size(0)
        # reshape audio to B x FRAMES x MFCC
        audio = audio.permute(0,2,1)

        if self.learn_states:
            _, [h_n, _] = self.encoder(audio, (self.ho.repeat(1,batch_size, 1), self.co.repeat(1,batch_size, 1)))
        else:
            _, [h_n, _] = self.encoder(audio)

        # concat the last layers' forward and backward states
        h_n = h_n.permute(1,0,-1)[:, -2:, :].reshape(batch_size, self.hidden_dim*2)
        embedding = self.projection_head(h_n)

        return embedding