'''
Goal: Use Viterbi to find the most probable sequence of rainy/not rainy given the sequence of observables (UCI dataset)

Dataset: AirQualityUCI.csv
 - Relevant features ['CO', 'NO2', 'O3']
 - data points in 1 hour intervals, ordered chronologically (our sequence of observables)

HMM:
 - Observables: ['CO', 'NO2', 'O3']
 - Hidden States: [0, 1] # [not rainy, rainy]
 - Transition Probs: [0.71428, 0.285714,
                     0.485981, 0.514019]
 - P(Rain | AQI_Category) = {1: 0.40404040404040403, 2: 0.3492063492063492, 3: None, 4: None} # {key=aqi_category, value=probability}, Note: None can be assumed to be 0
 
Steps:
1. Use CO, NO2, O3 to calculate AQI category for dataset, add as new column called 'AQI_Category'
    - 1 is 'Good', AQI of 0-50
    - 2 is 'Moderate', AQI of 51-100
    - 3 is 'Unhealthy_Sensitive', AQI of 101-150
    - 4 is 'Unhealthy', AQI of 151-200
    - 5 is 'Very_Unhealthy' 201-300
    - 6 is 'Hazardous' 301+
2. Generate rainy (1), not rainy (0) labels for dataset using transition probability and P(Rain | AQI_Category), add as new column called 'label'
3. Run Viterbi on HMM to generate most probable sequence of rainy/not rainy given the observables
4. Evaluate Viterbi sequence against generated labels
'''