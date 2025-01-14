import fetch from "node-fetch";

const df = {
    'columns': ['workclass', 'marital-status', 'occupation', 'education', 'race', 'sex', 'native-country', 'relationship', 'age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week'],
    'data': [
        ['State-gov', 'Never-married', 'Adm-clerical', 'Bachelors', 'White', 'Male', 'United-States', 'Not-in-family', 38, 10, 1077, 87, 40],
        ['State-gov', 'Never-married', 'Adm-clerical', 'Bachelors', 'White', 'Male', 'United-States', 'Not-in-family', 60, 30, 107700, 0, 60],
        ['State-gov', 'Never-married', 'Adm-clerical', 'Bachelors', 'White', 'Male', 'United-States', 'Not-in-family', 38, 10, 1077, 87, 40],
        ['State-gov', 'Never-married', 'Adm-clerical', 'Bachelors', 'White', 'Male', 'United-States', 'Not-in-family', 60, 30, 107700, 0, 60],
        ['State-gov', 'Never-married', 'Adm-clerical', 'Bachelors', 'White', 'Male', 'United-States', 'Not-in-family', 38, 10, 1077, 87, 40],
        ['State-gov', 'Never-married', 'Adm-clerical', 'Bachelors', 'White', 'Male', 'United-States', 'Not-in-family', 60, 30, 107700, 0, 60],
    ]
}

async function main() {
    const res = await fetch('http://localhost:8080/invocations', {
        method: 'POST',
        body: JSON.stringify({
            params: {method: 'proba'},
            dataframe_split: df,
        }),
        headers: {
            'Content-Type': 'application/json'
        },
    })
    const data = await res.json();
    console.log(data.predictions)
}

void main();