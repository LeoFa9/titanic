import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator, ClassifierMixin

# группы one-hot колонок (результат preprocessing.build_features), которые argmax'ом
# сворачиваются обратно в индекс категории для nn.Embedding
CATEGORICAL_GROUPS = {
    'Pclass': ['Pclass_1', 'Pclass_2', 'Pclass_3'],
    'Embarked': ['Embarked_C', 'Embarked_Q', 'Embarked_S'],
    'Title': ['Title_Master', 'Title_Miss', 'Title_Mr', 'Title_Mrs', 'Title_Rare'],
}
NUMERIC_COLS = ['Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'TicketGroupSize', 'HasCabin']


class EmbeddingMLP(nn.Module):
    def __init__(self, cat_cardinalities, embedding_dims, num_numeric, hidden_dim=16, activation=nn.Tanh):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(cardinality, dim) for cardinality, dim in zip(cat_cardinalities, embedding_dims)
        ])
        self.fc1 = nn.Linear(sum(embedding_dims) + num_numeric, hidden_dim)
        self.activation = activation()
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x_cat, x_num):
        embedded = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        x = torch.cat(embedded + [x_num], dim=1)
        return self.fc2(self.activation(self.fc1(x)))


class EmbeddingMLPClassifier(ClassifierMixin, BaseEstimator):
    """sklearn-совместимая обёртка над EmbeddingMLP, чтобы её можно было класть в VotingClassifier/StackingClassifier.
    ClassifierMixin обязательно первым в списке родителей - в sklearn 1.9 иначе is_classifier() возвращает False"""

    def __init__(self, embedding_dim=2, hidden_dim=16, lr=1e-3, n_epochs=50, batch_size=32, random_state=42):
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.random_state = random_state

    def _split_inputs(self, X):
        x_cat_cols = [X[cols].values.argmax(axis=1) for cols in CATEGORICAL_GROUPS.values()]
        x_cat = np.stack(x_cat_cols, axis=1)
        x_num = X[NUMERIC_COLS].values
        return torch.tensor(x_cat, dtype=torch.long), torch.tensor(x_num, dtype=torch.float32)

    def fit(self, X, y):
        torch.manual_seed(self.random_state)
        x_cat, x_num = self._split_inputs(X)
        y_arr = y.values if hasattr(y, 'values') else np.asarray(y)
        y_t = torch.tensor(y_arr, dtype=torch.float32).unsqueeze(1)

        cardinalities = [len(cols) for cols in CATEGORICAL_GROUPS.values()]
        self.model_ = EmbeddingMLP(
            cardinalities, [self.embedding_dim] * len(cardinalities), len(NUMERIC_COLS),
            self.hidden_dim, nn.Tanh
        )
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.lr)
        loss_fn = nn.BCEWithLogitsLoss()
        loader = DataLoader(TensorDataset(x_cat, x_num, y_t), batch_size=self.batch_size, shuffle=True)

        self.model_.train()
        for epoch in range(self.n_epochs):
            for xc, xn, yb in loader:
                optimizer.zero_grad()
                loss = loss_fn(self.model_(xc, xn), yb)
                loss.backward()
                optimizer.step()

        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        x_cat, x_num = self._split_inputs(X)
        self.model_.eval()
        with torch.no_grad():
            proba_1 = torch.sigmoid(self.model_(x_cat, x_num)).numpy().ravel()
        return np.column_stack([1 - proba_1, proba_1])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
