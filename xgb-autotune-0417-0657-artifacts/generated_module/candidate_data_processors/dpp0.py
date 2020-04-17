from numpy import nan
from sagemaker_sklearn_extension.externals import Header
from sagemaker_sklearn_extension.impute import RobustImputer
from sagemaker_sklearn_extension.preprocessing import RobustStandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

# Given a list of column names and target column name, Header can return the index
# for given column name
HEADER = Header(
    column_names=[
        'Churn', 'Account Length', 'VMail Message', 'Day Mins', 'Day Calls',
        'Eve Mins', 'Eve Calls', 'Night Mins', 'Night Calls', 'Intl Mins',
        'Intl Calls', 'CustServ Calls', 'State_AK', 'State_AL', 'State_AR',
        'State_AZ', 'State_CA', 'State_CO', 'State_CT', 'State_DC', 'State_DE',
        'State_FL', 'State_GA', 'State_HI', 'State_IA', 'State_ID', 'State_IL',
        'State_IN', 'State_KS', 'State_KY', 'State_LA', 'State_MA', 'State_MD',
        'State_ME', 'State_MI', 'State_MN', 'State_MO', 'State_MS', 'State_MT',
        'State_NC', 'State_ND', 'State_NE', 'State_NH', 'State_NJ', 'State_NM',
        'State_NV', 'State_NY', 'State_OH', 'State_OK', 'State_OR', 'State_PA',
        'State_RI', 'State_SC', 'State_SD', 'State_TN', 'State_TX', 'State_UT',
        'State_VA', 'State_VT', 'State_WA', 'State_WI', 'State_WV', 'State_WY',
        'Area Code_408', 'Area Code_415', 'Area Code_510', "Int'l Plan_no",
        "Int'l Plan_yes", 'VMail Plan_no', 'VMail Plan_yes'
    ],
    target_column_name='Churn'
)


def build_feature_transform():
    """ Returns the model definition representing feature processing."""

    # These features can be parsed as numeric.
    numeric = HEADER.as_feature_indices(
        [
            'Account Length', 'VMail Message', 'Day Mins', 'Day Calls',
            'Eve Mins', 'Eve Calls', 'Night Mins', 'Night Calls', 'Intl Mins',
            'Intl Calls', 'CustServ Calls', 'State_AK', 'State_AL', 'State_AR',
            'State_AZ', 'State_CA', 'State_CO', 'State_CT', 'State_DC',
            'State_DE', 'State_FL', 'State_GA', 'State_HI', 'State_IA',
            'State_ID', 'State_IL', 'State_IN', 'State_KS', 'State_KY',
            'State_LA', 'State_MA', 'State_MD', 'State_ME', 'State_MI',
            'State_MN', 'State_MO', 'State_MS', 'State_MT', 'State_NC',
            'State_ND', 'State_NE', 'State_NH', 'State_NJ', 'State_NM',
            'State_NV', 'State_NY', 'State_OH', 'State_OK', 'State_OR',
            'State_PA', 'State_RI', 'State_SC', 'State_SD', 'State_TN',
            'State_TX', 'State_UT', 'State_VA', 'State_VT', 'State_WA',
            'State_WI', 'State_WV', 'State_WY', 'Area Code_408',
            'Area Code_415', 'Area Code_510', "Int'l Plan_no", "Int'l Plan_yes",
            'VMail Plan_no', 'VMail Plan_yes'
        ]
    )

    numeric_processors = Pipeline(
        steps=[
            (
                'robustimputer',
                RobustImputer(strategy='constant', fill_values=nan)
            )
        ]
    )

    column_transformer = ColumnTransformer(
        transformers=[('numeric_processing', numeric_processors, numeric)]
    )

    return Pipeline(
        steps=[
            ('column_transformer', column_transformer
            ), ('robuststandardscaler', RobustStandardScaler())
        ]
    )


def build_label_transform():
    """Returns the model definition representing feature processing."""

    return LabelEncoder()
