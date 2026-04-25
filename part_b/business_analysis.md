# B1. Problem Formulation   
## (a) Machine Learning Formulation:    
This should be formulated as a supervised machine learning regression problem because the goal is to predict a numeric outcome: the `number of items_sold` for a store in a given month under a `specific promotion`.   

The `target variable` is items_sold.  (number of items sold by a store in a given month)    

The `candidate input features` include store-level attributes such as store size, location type, local competition density, monthly footfall, customer demographics, store segment, promotion type, calendar-related feature effects such as month, season, weekend effects, and festival periods.  

Regression is appropriate because the target variable (`items_sold`) is numeric and continuous and the business wants to estimate expected sales volume under different promotion choices, not just assign a class label. And once predicted sales are available for each promotion type, the company can choose the promotion with the highest expected sales for each store and month.

## (b) Why Items Sold Volume is Better Than Sales Revenue:  
Using items_sold is more reliable than total sales revenue because revenue can be influenced by `price differences`, `discount depth`, `premium products` and `inflation` so it may not reflect the true effect of a promotion on customer purchase volume. For example, a high-priced product may generate strong revenue even if only a few units were sold.  
Sales volume is a cleaner measure of promotion response because it is closer to the operational question the company is trying to answer: `which promotion increases customer purchase activity at a given store.`   

This illustrates a broader machine learning principle that the target variable should align closely with the real business decision the model is meant to support, rather than using a convenient but faulty metric as choosing the wrong target can optimise the wrong outcome and lead to poor business decisions. 

## (c) Alternative Modelling Strategy:  
Instead of one single global model, a better strategy would be a `hierarchical or segmented modelling strategy`.    
Such as for example, they can first be grouped into urban, semi-urban, and rural stores. Seperate models can then be trained for each segment, or a hierarchical model can be used with store-level effects, that includes interaction effects between promotion type and store characteristics, such as training models by location type or store cluster. 
    
Stores in different regions often have different customer behaviour, competition levels, purchasing power, and promotion responsiveness. A single global model may average these patterns and miss important local differences. Segment-based models allow predictions and promotion recommendations to better reflect each store environment, leading to higher accuracy and stronger business performance and feature importance analysis is specifically useful for identifying which variables drive predictions in different contexts.

# B2. Data and EDA Strategy 
## (a) Joining the Raw Tables and Final Dataset Design:   
The four raw tables would be combined using common keys such as `store_id`, `promotion_id`, and `transaction_date`. 

Join Process:   
**Transactions table**  
Use this as the base table because it contains sales outcomes such as items sold, transaction value, and dates.

**Store attributes table**  
Join using `store_id` to add store size, location type, footfall, competition density, and customer demographic information.

**Promotion details table**  
Join using `promotion_id` or promotion name to add promotion type, discount level, campaign duration, or promotion category.

**Calendar table**  
Join using `transaction_date` to add weekend flags, festival indicators, month, quarter, and seasonality variables.

The final modelling dataset should be:  
**One row = one store in one month under one promotion**    
This is because the business decision is which promotion to run at each store in each month 

At that level, useful aggregations would include:   
- Total items sold  
- Total revenue  
- Number of transactions  
- Average basket size  
- Average discount applied  
- Monthly footfall  
- Number of promotion days in the month  
- Percentage of weekend days  
- Festival count in the month   

## (b) Exploratory Data Analysis (EDA):   
Before building the model, I would perform the following analyses:  

Monthly Trend / Seasonality Analysis:    
A `time-series line chart` of `monthly items_sold` by store or by location type over `time` would help identify trend, seasonality, and festival spikes. If strong seasonality appears, it would justify creating time-based features such as month, quarter, or festival-period indicators.  

Promotion Performance Comparison:    
A `bar chart` or boxplot of `items_sold` by `promotion type` would Identify which promotions generally perform better and how variable their outcomes are. This would suggest strong-performing promotions may become key model signals, while weak or unstable promotions may need further segmentation.   

Sales by Store Location Type:   
A `heatmap` or `grouped bar chart` of `average items_sold` by `promotion type` and `location type` would show whether the same promotion behaves differently in urban, semi-urban, and rural stores. If the effect differs a lot by location, that would support interaction features or segmented models.  

Correlation / Scatter Analysis: 
A `Correlation heatmap` or `scatter plot` for variables such as `footfall, competition density, and items_sold`. This measures relationships between numerical variables and sales. This would influence decisions such as scaling, transformations, or the use of robust models.

## (c) Impact of 80% No-Promotion Imbalance:    
If 80% of observations occurred without promotions, the model may become biased toward learning normal non-promotion behaviour and underestimating the effect of promotional campaigns. As a result, due to this imbalance predictions for promotion scenarios may be less accurate for the smaller set of promoted transactions and may lead to poor recommendations when comparing promotion options. 
To address this, the data should be evaluated separately for promoted and non-promoted cases, promotion-level performance should be checked by segment, and balancing strategies such as weighting, resampling at the aggregated level, or using evaluation metrics that focus on minority cases should be considered. Also collection of historical promotion and casual methods to estimate incremental promotion impact should also be evaluated.    

# B3. Model Evaluation and Deployment   
## (a) Train-Test Split and Evaluation Metrics: 
Since the data is monthly and spans three years, I would use a **time-based split** rather than a random split.     

The model should be trained on the earlier months and tested on the most recent months, for example using the first two to two-and-a-half years (30 months) for training and the final 6 months as the test set.    
This ensures the model is trained on historical data and evaluated on future unseen months.

A random split is inappropriate because it would mix future observations into the training data, creating data leakage and making the model appear more accurate than it would be in real use.      

Because the target is `items_sold`, this is a regression problem. The main evaluation metrics should be: 

**RMSE** (Root Mean Squared Error): 
Measures average prediction error while penalising large mistakes more heavily. Useful when large forecasting errors could cause stock shortages or overstocking.   

**MAE** (Mean Absolute Error):  
Measures average absolute units missed. Easy for business users to interpret (e.g., “the forecast is off by 12 items on average”).  

**R² Score:**   
Measures how much variation in sales is explained by the model. Higher values indicate stronger predictive power.       

**Promotion Recommendation Accuracy (Business Metric):**    
Track whether the model-selected promotion outperformed the historical alternative. This directly measures decision quality rather than only prediction accuracy.           

## (b) Explaining Different Recommendations for December vs March:  
The model recommends different promotions because the drivers of demand differ across months. I would investigate using feature importance and explainability tools (such as partial dependence). I would compare the main factors influencing predictions for Store 12 in December versus March.   

Likely December drivers:
- Festival or holiday season  
- Higher customer demand  
- Gift-buying behaviour  
- Higher footfall   

These conditions may make **Loyalty Points Bonus** more effective because customers are already purchasing and loyalty incentives increase basket size or repeat visits.    

Likely March drivers:   
- Lower seasonal demand  
- Price-sensitive customers after holiday spending  
- Normal traffic levels 

These conditions may make **Flat Discount** more effective because immediate price reductions encourage purchases.  

Feature importance helps identify which variables the model relies on most overall, while local explanation for the two predictions can show which features pushed the December prediction toward one promotion and the March prediction toward another.    
This should be communicated to the marketing team by explaining that the recommendation is based not only on the store itself but also on the month-specific context, so the same store can receive different recommendations when seasonal or promotional conditions change.   

## (c) Deployment Process and Monitoring:   
For deployment, the complete preprocessing and modeling pipeline should be saved after training, for example using joblib, so that the same encoders, scalers, and model logic can be reused every month without rebuilding everything from scratch.    

At the start of each month, new data from stores, promotions, and the calendar should be collected, transformed into the same feature structure used during training, and passed into the saved pipeline to generate predicted items_sold for each promotion option for all 50 stores.  

The chosen recommendation for each store would then be the promotion with the highest predicted sales volume.   

Monitoring should include tracking actual versus predicted sales once the month ends, checking data drift in important features, watching for changes in promotion mix or store behavior, and setting retraining triggers when model error rises or input data shifts significantly from the original training distribution.

