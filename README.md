# Climate-Driven Forecasting of Wheat Pathogen Outbreaks  
**High School Big Data & AI Challenge 2025–2026**

**Author:** Gurnoor Kaur  
**School:** Central Peel Secondary School  
**Date:** January 2026  

## Project Overview
Climate variability is increasingly influencing the spread of crop pathogens, making early detection critical for preserving global food security. In wheat production, short term fluctuations in climate and soil anomalies content during key growth stages substantially affect the nature of harvest and susceptibility to pathogen outbreaks. 

This study analyzes climate and soil parameters and pathogen datasets from 1995 to 2024, using both statistical and machine learning models. Pearson correlation analysis and linear regression was applied to asses relationships between both individual and linked environmental variables and pathogen impact. A Random Forest predictive model was developed to distinguish between outbreak and non outbreak years. The results identified moisture related variables such as precipitation, humidity, vapour pressure deficit, and soil moisture to have the most significant influence on pathogen occurrence and impact. The Random Forest model achieved an accuracy of up to 78\% thus showing a strong capibility in identifying outbreak years across both Canadian and U.S. regions.

These findings highlight the effectiveness of incoporating climate and soil analytics in predictive machine learning models to create early warning systems for pathogen outbreaks. By taking proactive risk evaluation measures, based upon the results from such tools, we can strengthen agricultural preparedness, reduce yield losses, enable policy makers make more informed trade decisions and enhance global food security.

## Research Questions

1. To what extent do temperature, soil moisture, precipitation, humidity, and vapour pressure deficit (VPD) impact wheat pathogen outbreaks?  
2. Can climate and soil datasets be used by machine learning models to accurately predict outbreak years?

### Climate & Soil  
- **Source:** Open-Meteo API  
  - Canada: 1995-2024  
  - USA: 2000-2024  

### Pathogen Data  
- **Canada:**  
  - Source: Canadian Grain Commission  
- **United States:**  
  - Source: USDA ARS Cereal Rust Bulletins  
### Market Data (USA)  
- Yield & Price: USDA NASS  
- Imports: USDA Foreign Agricultural Service  

**Statistical Analysis**  - Pearson correlation & Linear regression  
**Machine Learning**  
   - Random Forest classification  
   - Canada: Train 1995–2018, Test 2019–2024  
   - USA: Train 2000–2019, Test 2019–2024  

## HOw to run:
Install the required Python libraries (pandas, numpy, matplotlib, scikit-learn, scipy, and statsmodels), then run the analysis codes in models/canada and models/us for time series, Pearson correlation, linear regression, and Random Forest classification. All datasets used in the models are stored in the data folder. 
