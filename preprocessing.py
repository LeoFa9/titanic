import pandas as pd
from sklearn.preprocessing import StandardScaler

CATEGORICAL_COLS = ['Pclass', 'Embarked', 'Title']
SCALE_COLS = ['Age', 'Fare', 'SibSp', 'Parch', 'TicketGroupSize']


def extract_title(names):
    """Достаёт титул (Mr/Mrs/...) из полного имени, редкие варианты группирует в Rare"""
    title = names.str.extract(r' ([A-Za-z]+)\.')[0]
    title = title.replace({'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs'})
    common_titles = {'Mr', 'Mrs', 'Miss', 'Master'}
    return title.where(title.isin(common_titles), 'Rare')


def engineer_features(df, ticket_counts):
    """Строит фичи Title/TicketGroupSize/HasCabin из Name/Ticket/Cabin и удаляет исходные текстовые колонки"""
    df = df.copy()
    df['Title'] = extract_title(df['Name'])
    df['TicketGroupSize'] = df['Ticket'].map(ticket_counts).fillna(1)
    df['HasCabin'] = df['Cabin'].notna().astype(int)
    return df.drop(['Name', 'Ticket', 'Cabin'], axis=1)


def preprocess_data_advanced(df, is_train, artifacts=None):
    """Полный препроцессинг: генерация фичей + заполнение пропусков.
    На train (is_train=True) считает статистики и возвращает их в artifacts,
    на test (is_train=False) использует artifacts, посчитанные на train"""
    df = df.drop(['PassengerId'], axis=1)

    if is_train:
        artifacts = {'ticket_counts': df['Ticket'].value_counts()}

    df = engineer_features(df, artifacts['ticket_counts'])

    if is_train:
        artifacts['mode_embarked'] = df['Embarked'].mode()[0]
        artifacts['median_age_by_title'] = df.groupby('Title')['Age'].median()
        artifacts['median_fare'] = df['Fare'].median()

    df['Embarked'] = df['Embarked'].fillna(artifacts['mode_embarked'])
    df['Fare'] = df['Fare'].fillna(artifacts['median_fare'])
    df['Age'] = df['Age'].fillna(df['Title'].map(artifacts['median_age_by_title']))
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

    return (df, artifacts) if is_train else df


def build_features(train_df, test_df):
    """Полный цикл фичей для обучения: preprocess_data_advanced + one-hot категорий + масштабирование.
    Тот же зафиксированный пайплайн, что использовался во всех notebook-экспериментах проекта"""
    test_passenger_ids = test_df['PassengerId']

    train_df, artifacts = preprocess_data_advanced(train_df, is_train=True)
    test_df = preprocess_data_advanced(test_df, is_train=False, artifacts=artifacts)

    train_df = pd.get_dummies(train_df, columns=CATEGORICAL_COLS)
    test_df = pd.get_dummies(test_df, columns=CATEGORICAL_COLS)
    test_df = test_df.reindex(columns=train_df.drop('Survived', axis=1).columns, fill_value=0)

    scaler = StandardScaler()
    train_df[SCALE_COLS] = scaler.fit_transform(train_df[SCALE_COLS])
    test_df[SCALE_COLS] = scaler.transform(test_df[SCALE_COLS])

    X_train = train_df.drop(['Survived'], axis=1)
    y_train = train_df['Survived']
    X_test = test_df

    return X_train, y_train, X_test, test_passenger_ids
