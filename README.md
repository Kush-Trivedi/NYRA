<center><img src='https://cdn.dribbble.com/users/819689/screenshots/6176353/gallop.gif' width=800></center>

<h1 style="color:#EE2737FF"><center><font size="+6">From Canter -To- Gallop</font></center></h1>

<h3 style="text-align: right;color:rgb(0, 103, 71);"><i>By: Kush Trivedi</i></h3>

This **notebook** seeks to **illustrate** the **likelihood** for each **jockey  that could win a race** with it's respective position based on **significant race variables**, **track conditions**, and **in particular the odds that are currently being placed** on each **individual horse / jockey**. This notebook will also be most helpful to **horse owners**, **trainers**, and **general people**. Not only but also at last we define a math model to find the **solid wager for long term profits** that will increase in direct proportion to the number of wagers you **place** and **win**.

Mainly, the **notebook** has **5 Parts** with an **example** of one small **sample race** for understanding purpose:
1. [**Part 1:** Finding an **Optimum Path**.](#001)
2. [**Part 2:** **Visualizing** and **Understanding** few **typical scenerio** that happens **during the race**](#002)
3. [**Part 3:** Showing **Statical Siginificant diffrence** for the **Optimum Path** and **Current Path** by performing **A/B testing** with **Two Related Sample Test**.](#003)
4. [**Part 4:** **Evalution Metric** Summary of an **Optimum Path**](#004)
5. [**Part 5:** Addressing the **issues found in the Evaluation Metric Summary** and **Defining** the **Probability Estimate Model** to **forecast** each jockeys **Probable Place** based on **Odds**.](#005)

* [**Notebook with All Detail Explaination**](https://www.kaggle.com/code/kushtrivedi14728/advance-eda-with-ml-big-data-derby)
* [**Source Code**](https://github.com/Kush-Trivedi/NYRA)


<h2 style="text-align: center;color:rgb(0, 103, 71);"><b id="001">Part 1: Finding an Optimum Path.</b></h2>

In **simple words** there can be **N number of paths** in a **N number of tracks** also there are **N number of maths**, **physics** and **personal reasearch** done by **N number of competitors** in this **competition**. However, the question is **what could be the best optimum path?** Simply, In **my personal opinion** I would say that **sometimes implementing simple logic is better than using complex theory or partical concept** by this, I mean that the **jockey who won the race would have undoubtedly followed the best route**, **strategy**, and **other tactics**. Therefore, by **extracting data** from various **race variables**, we **may identify** the **Optimum Path**. For an instance extract data by **track_id**, **furlongs**, **course_type**, **track_condition**, and **finishing_place**.

**Contrarily**, In the views of the **other people** and **competitors** can state that **each horse is unique**, with some taking **large stides / gait pattern** and others taking **smaller** ones. Some jockeys **may have achieved victories through pure chance** or other **unforeseeable circumstances**. So people **may debate on my findings for optimum path strategy** however I can prove them with **statical significant test**.

So, after **carefully examining** of the **derby coordinates data** for the **victorious jockeys (1st Place)**, I discovered that the **data is not distributed symmetrically it's a skewed distribution**, making it a **bad idea to take the Mean** coordinates to get Optimum Path. **Instead, we should take the coordinates Median**, and not only but also when we have a skewed distribution, the **median is a better measure of central tendency than the mean** also by taking Median it will help us **avoid the contrary situation** mentioned above.


<h3 style="color:#EE2737FF">Funtion to Find an Optimum Path</h3>

```python

# Load Tracking Data
complete = pd.read_csv('/Users/kushtrivedi/Desktop/NYRA/nyra_2019_complete.csv',low_memory=False)

# Define Function and it's parameters. 
# This Function is developed by Kush Trivedi Ⓒ
def find_optimum(track_id:str,furlongs:float, course_type:str,track_condition:str,finishing_place:int):
    
    # Extract Data on various Race Vaiables
    extracted_df = complete.query(
        "track_id == @track_id & furlongs == @furlongs & course_type == @course_type & track_condition == @track_condition & \
        finishing_place == @finishing_place"
    ).sort_values("trakus_index")
    
    # Extraxt Required Columns
    result = extracted_df[[
        "track_id","race_date","race_number","run_up_distance","jockey",
        "program_number","horse_name","furlongs","trakus_index","latitude","longitude"
    ]]
    
    # Takes Median of Latitude and Longitude
    average_lat = pd.DataFrame({'average_lat':result.groupby('trakus_index')['latitude'].median().astype(float)}).reset_index()
    average_longi = pd.DataFrame({'average_longi':result.groupby('trakus_index')['longitude'].median().astype(float)}).reset_index()
    
    # Merge the Optimum Coordinates and return the final dataframe
    final_average = pd.merge(average_lat, average_longi, on=["trakus_index"])
    final_average['optimum_jockey'] = 'Optimum Path'
    final = pd.merge(result,final_average,on=["trakus_index"])
    return final

# An Instance of Finding an Optimum Path by providing crucial Race Varaibles parameters to the function
sar_six_furlong_in_dirt_with_fast_condition_optimum_path = find_optimum("SAR", 6, "D", "FT ", 1)

# Another Instance
aqu_six_furlong_in_dirt_with_fast_condition_optimum_path = find_optimum("AQU", 6, "D", "FT ", 1)

```

**Similarly**, we can find **Optimum Path for other combinations**. So, **once** we have **identified the Optimum Paths** lets **Visualize** one race first to see **if we are correct or not** and than by performing **Statical Test** and **Polynomial Regression with Evalution Matric**.

<h2 style="text-align: center;color:rgb(0, 103, 71);"><b id="002">Part 2: Visualizing and Understanding few Typical Scenerio that Happens During the Race</b></h2>

Here we will analyze the Race that happned at **Saratoga on 27th July 2019** for **understanding purpose** as this **race has most common scenerio** that **happens during ongoing race** and we will visulize it step by step with our Optimum Path coordinates. **Video** of race can be find [**here**](https://youtu.be/-OId_SqGt_I?t=2142)

<h3 style="color:#EE2737FF">Race Summary</h3>

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F8412510%2Feda78a7241a08aff64f3bc9b521d7c51%2FFinal_Sar_Summary.png?generation=1666467906484458&alt=media)

From the above summary, it is clear that **higher average speed favors a winning position**, **but are we convinced** that **higher average speed will always put us in a winning position?** Well, we can **agree** with that statement **to a certain extent**. So, let's examine the **top 3 jockeys** performances. Here, we can observe **a type of scenerio** in which **Joel Rosario**, who **finished first**, had a **higher average speed**. **However**, if we examine the race summary for **Javier Castellano**, we find that he **placed second with a slightly lower average speed** than **Manuel Franco, who finished third with a slightly higher average speed**. **Strange, huh?** So now let's take a closer look at what might have gone wrong in the animation below.

<h3 style="color:#EE2737FF">Top 3 Jockey Comparision with an Optimum Path </h3>

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F8412510%2F1964d5604572d85faaaae124899d70d1%2Ffinal_all.gif?generation=1667443953079954&alt=media)

As in the animation above, if **we look closely**, we can observe a **ranking difference between** the **jockeys** for **every four dashes**. I know it hard to understand and **waste of time to look at animation for few seconds**. Therefore, let's **simplify** it by providing several key **speed-time concepts** together with their typical **curves** to have a more in-depth comprehension.

<h3 style="color:#EE2737FF">Typical Speed Time Curve</h3>

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F8412510%2F8aba1981983c84423c7579bcb340233e%2FSpeed-time-curve.jpeg?generation=1665796320262949&alt=media)

In essence, a typical speed time curve shows that traveling **from point 0 to A in 0 to t1 time** is referred to as a **rehostatic acceleration** (**Constant Acceleration**). This is frequently the **starting line of a race**, and somewhat **staright path**. We can now see a curve forming **from point A to point B**, which depicts that **speed on the curve**; **B to point C** is known as **free running on the curve**; **C to point D** is known as **coasting**; and, finally, **point D to point E** is a **braking point**, where body speed and acceleration are reduced to take a **wider** or **shorter turn**. Now, lets look the speed time curve for top 3 jockey.

<h3 style="color:#EE2737FF">Speed Time Curve of Top 3 Jockey</h3>

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F8412510%2F110b848c959f5fb26f3b8f44a610760b%2Fjoel_vs_jaiver_vs_manuel.png?generation=1667445005224393&alt=media)

<h3 style="color:#EE2737FF">Acceleration at Various Points on Race Track</h3>

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F8412510%2Fe4464de08d7f44904e28dca61f7ae604%2Fjoc_acc_on_track.png?generation=1667447093080173&alt=media)

The visualization shown above makes it very **evident** that **Joel Rosario** had **strong acceleration** following the braking point and **maintained his lead** throughout the remainder of the race. Furthermore, if we compare **Javier Castellano** and **Manuel Franco**, we can see that **Manuel took the lead at curvature** since it was clear that Manuel would get benefit at the curve as he was **taking wider turn** and **apexing late**. However, **Javier** accelerated at the **exit point** and **took** the **lead** after **Manuel made a mistake** at the **exit point**, which caused him to loose the race. **Simple Right? Answer is NO**. this could be **one scenerio** but there **might be multiple scenerio** and **unforeseeable circumstances**. So, let's look at that by comparing with optimum path data.

<h3 style="color:#EE2737FF">Speed Time Curve of an Optimum Path</h3>

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F8412510%2F7999f2c8e93266b676ea248efaf218c0%2Foptimum_six_on_dirt_sar.png?generation=1667448393754638&alt=media)

<h3 style="color:#EE2737FF">Comparision of Speed Time Curve of an Optimum Path with Top 3 Jockey</h3>

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F8412510%2F33fd50c741a6e5bac4dbebd80676518d%2Foptimum_six_on_dirt_sar_mix_all.png?generation=1667445015627506&alt=media)

<h3 style="color:#EE2737FF">Spohisticated Speed Time Curve with Polynomial Regression</h3>

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F8412510%2Fd88278d7f26f6a25682f0b7381243a34%2Fjoc_with_optimum_reg.png?generation=1667501698321902&alt=media)

The above visualization clearly demonstrates that a **curve line that is closer to the optimum curve line will result in a winning position** in the race. For example, the speed time curve for **Joel Rosario follows the optimum path**, while the speed time curve for **Javier** and **Manuel** initially **shows Javier in the lead at the turn-in point** but **Manuel overtook him after taking a wider turn on the curve**, but again, if we look at the exit point curvature, **Javier regains the lead**.

**However**, though it can be clearly seen in visualization there are **multiple factor are playing their role**. First, **is Manuel Horse Older than Javier Horse? Second, is Javier a Good Jockey compare to Manuel and vice vera? Third, is the trainer and owner are good in multiple scenerio for each individaul horse and jockey?** and much more other scenerios. So it's hard to say with this type of data provided. However, I have **performed** an **statical test to prove it** temporarily.


<h2 style="text-align: center;color:rgb(0, 103, 71);"><b id="003">Part 3: Statical Testing</b></h2>

What is the purpose of statical testing, first and foremost? **To determine whether there is a difference between a jockey's current path and an optimal path? By taking an optimum path will help the Jockey to win race?**

This Kind of Questions can be solved with **t-test two related sample test**

First we will **extract** the **required data** shown below:

```python

# Optimum Coordinates
df_opt = t_test_stat[(t_test_stat['is_opt'] == 1)][
    ["Speed (Mph)","Optimal Speed (Mph)","Acceleration","Optimal Acceleration","is_opt"]
]

# Not Optimum Coordinate
df_not_opt = t_test_stat[(t_test_stat['is_opt'] == 0)][
    ["Speed (Mph)","Optimal Speed (Mph)","Acceleration","Optimal Acceleration","is_opt"]
]

```

Now let's define the **Null Hypothesis** and **Alternative Hypothesis** for **Speed** and **Acceleration**

* <h3 style="color:#EE2737FF">Speed</h3>

    * Jockey who were **Not on an Optimum Speed** w.r.t to track_us_index

        - **Null Hypothesis**: among jockeys who wasn't on an optimum speed, the speed of the current path is the same as the optimum path.

        - **Alternative Hypothesis**: among jockeys who was on an optimum speed, the speed of the optimum path is better than the current path.

    * Jockey who were **on an Optimum Speed** w.r.t to track_us_index

        * **Null Hypothesis**: the average speed before and after taking an optimum path is the same.

        * **Alternative Hypothesis**: the average speed after taking an optimum path (Optimum Speed (Mph)) is better than before (Speed (Mph)).

    * Notice that with the way we formulate the alternative hypothesis, we’re conducting a **left-sided hypothesis**.


```python

test_statistic, p_value = ttest_rel(df_not_opt['Speed (Mph)'], df_not_opt['Optimal Speed (Mph)'],alternative='less')

```

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F8412510%2Fc05c1b02dde9bc97556161f62ae75a06%2FScreen%20Shot%202022-11-09%20at%2012.48.11%20AM.png?generation=1667976695595923&alt=media)

As we can see, the **p-Value is very small**, which means that we can say that the **jockeys path after taking an optimum path is indeed better than current path**. At a significance level of 0.05, **we reject the null hypothesis in favor of the alternative hypothesis**.


```python

test_statistic, p_value = ttest_rel(df_opt['Speed (Mph)'], df_opt['Optimal Speed (Mph)'],alternative='less')

```

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F8412510%2F1005855dae4ebae90e1890a8ccce6012%2FScreen%20Shot%202022-11-09%20at%2012.48.20%20AM.png?generation=1667976755017696&alt=media)


* <h3 style="color:#EE2737FF">Acceleration</h3>

    * Jockey who were **Not on an Optimal Acceleration** w.r.t to track_us_index

        - **Null Hypothesis**: among jockeys who didn’t applied an optimal acceleration, the acceleration applied on the current path is the same of the optimum path.

        - **Alternative Hypothesis**: among jockeys who didn't applied an optimal acceleration, the acceleration applied on the optimum path is better than the current path.

    * Jockey who were on an **Optimum Acceleration** w.r.t to track_us_index

        * **Null Hypothesis**: the average acceleration applied before and after taking an optimum path is the same.

        * **Alternative Hypothesis**: the average acceleration on an optimum path (Optimal Acceleration) is better than that of current path (Acceleration).
        
    * Notice that with the way we formulate the alternative hypothesis, we’re conducting a **left-sided hypothesis**.

```python

test_statistic, p_value = ttest_rel(df_not_opt['Acceleration'], df_not_opt['Optimal Acceleration'],alternative='less')

```

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F8412510%2F2a2c8eacddd3ae9c41a3262ee3605124%2FScreen%20Shot%202022-11-09%20at%2012.57.16%20AM.png?generation=1667977663223196&alt=media)


```python

test_statistic, p_value = ttest_rel(df_opt['Acceleration'], df_opt['Optimal Acceleration'],alternative='less')

```

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F8412510%2F945ce753b728ebb69cf3744c204e4a4d%2FScreen%20Shot%202022-11-09%20at%2012.59.30%20AM.png?generation=1667977354409268&alt=media)


<h2 style="text-align: center;color:rgb(0, 103, 71);"><b id="004">Part 4: Evalution Metric Summary of an Optimum Path</b></h2>

So as we have proved that Optimum Path is better than the Current Path we can now evalaute and calculate the **result** of the **actual** and the **optimum speed**. Bellow is the **Evalution Metric Summary** for each jockey for that particular race.

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F8412510%2F9bbf5bab0118a4ef7eb2cdc58dea3b4d%2FEvalution.png?generation=1667501332826574&alt=media)

As we had previously visualized, that the **accuracy is improved by being closer to the optimum curve line**. We also demonstrated that the **optimum path speed is preferable than the current speed**, and this is also evident in the evaluation metric summary. **However**, we are aware that a variety of variables, including the **horse's age**, the **jockey's expertise**, the **trainers** and **owners**, and many more **other variables**, may have a **significant impact**. As a result, here we can also see one more interesting scenerio that **though Jose Ortiz** followed the **ideal path according to our evalution metric**, why did he **finished in sixth place?** Let's examine it and lets find way to overcome this problem.

<h3 style="color:#EE2737FF">Comparision of Speed Time Curve of Joel Rosario and Jose Ortiz</h3>

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F8412510%2F46cfca95e135f075e52d2b9e93554e2c%2Fjr_jo_reg.png?generation=1667986865283589&alt=media)

Here we can clearly see that **Jose Ortiz also follows** the same **optimum path coordinates** however his **speed was less compare to Joel Rosario**. So sometimes this **kind of situation arries** and **due to that it might predict wrong result**. So how can we overcome this situation? Answer is very simple we need to come with **our own model** instead of using **Sklearn** or **third party library** and we need to **make** some **data inspired decisions** and by **levaraging** the **use of external dataset** which include all of the crucial race varaibles that are not currently present in provided dataset.

<h2 style="text-align: center;color:rgb(0, 103, 71);"><b id="005">Part 5: Probability Estimate Model to Forecast Each Jockeys Probable Place based on Odds</b></h2>

As we have seen that all above visualizations there were **some factors** that are **causing our model to predict wrong** also the provided dataset does not have all the main crucial data that play an important role to predict the final position of jockey. So, I decided to make my own dataset that is available at [Equibase](https://www.equibase.com/) website at not cost and then by creating a Mathemetical Equation to predict the probability for each individual shown bellow.

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F8412510%2F69e0f3a8728ab93bce1fcc1c2c763b51%2FMy_Math.png?generation=1668070767962674&alt=media)

Now, lets breakdown our math model to Predict the Jockey / Horse Chance of Winning in simple 3 Steps and finding the solid wager.

* **Step 1**: Find out the Percentage Chance that Racebook thinks each Horse will by applying Implied Probability.
* **Step 2**: Fetch your Custom Ratings in scale of 10 for each Name Entity(Jockey, Trainer, Owner, Weather, Horse Age etc).
* **Step 3**: Assign the value for each Name Entity to our Math Model to get final results.

By performing these 3 steps it will help us get the percentage chance winning for each horse and lets break down each step in detail with the example  and lets analayze and predict the results for the race that is going to be happened on **Nov 6, 2022 at AQU**.

<h3 style="color:#EE2737FF">Step 1: Find Implied Probablity</h3>

```python

# Function to Find Implied Probability

# If the Odds are in Decimal
def decimal_odds(odds):
    implied_probability = round((1 / odds) * 100 ,2)
    return implied_probability

# If the Odds are in Fraction
def fractional_odds(numerator,denominator):
    implied_probability = round(denominator / (denominator + numerator) * 100,2)
    return implied_probability

# If the Odds are Negative
def negative_american_odds(odds):
    implied_probability = round(odds / (odds + 100) * 100 ,2)
    return implied_probability

# If the Odds are Positive
def positive_ameriacn_odds(odds):
    implied_probability = round((100 / (odds + 100)) * 100,2)
    return implied_probability

```

But, before finding the implied probability lets understand **what are the odds** in the derby race. **Odds** are nothing but it **tells how much we stand to win if we bet on that particular horse**. For example let's predict the race that is going to be happened on Nov 6, 2022 with my personal dataset that I have created. So **we always get the odds value before the race starts** for an instace lets look at odds that were placed for **AQUEDUCT - November 6, 2022 - Race 1** where **race type** is **MAIDEN CLAIMING** there were **7 Horses** competing with each other with there odd value placed mentioned bellow.

* Horse 1: **Tall Girl** with an odds placed of **2.10**
* Horse 2: **Words of Praise** with an odds placed of **4.60**
* Horse 3: **Here We Go Again** with an odds placed of **31.00**
* Horse 4: **Addressable Market** with an odds placed of **3.45**
* Horse 5: **Nightsaber** with an odds placed of **15.00**
* Horse 6: **Welcometomyworld** with an odds placed of **24.75**
* Horse 7: **Cookie Crumbs** with an odds placed of **1.95**

We can see that the **odds value are in decimal** so we can simply put those value in **our equation** above to get the **Implied probability** **for** each horce that **racebook** thinks are likely to win and after finding the Implied probability it would look somewhat like shown bellow.

* **Tall Girl** implied probablity is **47.62 %**
* **Words of Praise** implied probablity is **21.74 %**
* **Here We Go Again** implied probablity is **3.23 %**
* **Addressable Market** implied probablity is **28.99 %**
* **Nightsaber** implied probablity is **6.67 %**
* **Welcometomyworld** implied probablity is **4.04 %**
* **Cookie Crumbs** implied probablity is **51.28 %**

Now we got the **initial percentage chance** for each horse that racebook thinks that they could win. **Pretty Neat, Correct?** But why its is important for us. Well **racebook pay us out based on the likelihood of your bet winning**. If a **horse is more likely to win** the race you are **not** going to get **paid out** as **much** because that's pretty likely to happen more often and vice versa for the **underdog horses**. So our **main goal is to look for the horse that we think are more likely to win with our own custom rating value** and **compare them with the racebook percentage**. But how can we get the custom rating value?



<h3 style="color:#EE2737FF">Step 2: Find Custom Rating Value for each Name Entity</h3>


```python

# Define Custom Rating 
custom_rating_map = {
    1: '10', 2: '9', 3: '8', 4: '7', 5: '6', 
    6: '5', 7: '4', 8: '3', 9: '2', 10: '1', 
    11: '1', 12: '1', 13: '1', 14: '1', 15: '1', 
    16: '1', 17: '1', 18: '1', 19: '1', 20: '1'
}
df['rating'] = df['finish_position'].map(custom_rating_map)


# Get Jockey Custom Rating
jockey = pd.DataFrame({'Frequency': df.groupby(['jockey','rating'])['rating'].nunique()}).reset_index()
jockey_rating_overall = pd.DataFrame({'overall_rating':jockey.groupby('jockey')['rating'].median().round().astype(int)}).reset_index()
jockey_last_20_day = pd.DataFrame(
    {
        'Frequency': last_20_day_df.groupby(
            ['race_date', 'race_tyoe','distance_id','course_type','track_condition','jockey','finish_position']
        )['finish_position'].nunique()
    }
).reset_index()
jockey_last_20_day['rating_update'] = jockey_last_20_day['finish_position'].map(custom_rating_map)
jockey_rating_updated = pd.DataFrame({'rating_update':jockey_last_20_day.groupby('jockey')['rating_update'].median().round().astype(int)}).reset_index()
jockey_rating = jockey_rating_overall.merge(jockey_rating_updated, how='left').fillna(0)
jockey_rating['rating'] = ((jockey_rating['overall_rating'] + jockey_rating['rating_update']) / 2)
jockey_rating = jockey_rating[["jockey","rating"]]


# Get Trainer Custom Rating
trainer = pd.DataFrame({'Frequency': df.groupby(['trainer','rating'])['rating'].nunique()}).reset_index()
trainer_rating = pd.DataFrame({'rating':trainer.groupby('trainer')['rating'].median().round().astype(int)}).reset_index()


# Get Owner Custom Rating
owner = pd.DataFrame({'Frequency': df.groupby(['owner','rating'])['rating'].nunique()}).reset_index()
owner_rating = pd.DataFrame({'rating':owner.groupby('owner')['rating'].median().round().astype(int)}).reset_index()


# Get Program Number Custom Rating
program_number = pd.DataFrame({'Frequency': df.groupby(['program_number','rating'])['rating'].nunique()}).reset_index()
program_number_rating = pd.DataFrame({'rating':program_number.groupby('program_number')['rating'].median().round().astype(int)}).reset_index()


# Get Horse Age Custom Rating
horse_age_data = {'age': [2,3,4,5,6,7,8,9,10,11],'rating': [6,10,9,8,7,5,4,3,2,1]}
horse_age_rating = pd.DataFrame(horse_age_data)


# Get Weather Custom Rating
# Weather Rating will be constant for all horse as the race will happen on same day. (We can make it more sophisticated by taking range of temprature and then assign the rating value for each sub category weather)

```

Once we get the **Custom Rating values for each Name Entity** we can now **assign it to our math model** to get the percentage chance winning.


<h3 style="color:#EE2737FF">Step 3: Math Model to get final results</h3>

```python

# Assign the percentage weight for each name entity
def predict(jockey,trainer,owner,program_number,horse_age,weather):
    value = (0.40 * jockey) + (0.30 * trainer) + (0.10 * owner) + (0.10 * program_number) + (0.05 * horse_age) + (0.05 * weather)
    return value

```

So, after applying the percentage weight we will get the value for each horse / jockey and lastly divide each horse / jockey value with total value.


```python

# Assign your own custom rating value 
def final_predict(j1,j2,j3,j4,j5,j6,j7,t1,t2,t3,t4,t5,t6,t7,o1,o2,o3,o4,o5,o6,o7,a1,a2,a3,a4,a5,a6,a7,w1,w2,w3,w4,w5,w6,w7,p1,p2,p3,p4,p5,p6,p7):
    first = predict(j1,t1,o1,a1,w1,p1)
    second = predict(j2,t2,o2,a2,w2,p2)
    third = predict(j3,t3,o3,a3,w3,p3)
    fourth = predict(j4,t4,o4,a4,w4,p4)
    fifth = predict(j5,t5,o5,a5,w5,p5)
    sixth = predict(j6,t6,o6,a6,w6,p6)
    seven = predict(j7,t7,o7,a7,w7,p7)
    
    total = first + second + third + fourth + fifth + sixth + seven
    
    first_final = round((first / total) * 100,2)
    second_final = round((second / total) * 100,2)
    third_final = round((third / total) * 100,2)
    fourth_final = round((fourth / total) * 100,2)
    fifth_final = round((fifth / total) * 100,2)  
    sixth_final = round((sixth / total) * 100,2) 
    seven_final = round((seven / total) * 100,2) 
    
    return [[first_final], [second_final], [third_final], [fourth_final], [fifth_final], [sixth_final], [seven_final]]

```

You now possess all values and **compare the implied probability you obtained from the racebook with the percentages you believe the horse needs to win**. If your **winning % exceeds that of the racebooks, you have probably found a solid wager**, providing you are right. Long-term profits will increase in direct proportion to the number of wagers you place and win.

So after applying our custom ratings we got percentage such as:

* We found **Tall Girl** has **19.92 %** winning chnace,and the implied probablity was of **47.62 %**
* We found **Words of Praise** has **18.04 %** winning chnace,and the implied probablity was **21.74 %**
* We found **Here We Go Again** has **16.16 %** winning chnace,and the implied probablity was **3.23 %**
* We found **Addressable Market** has **14.29 %** winning chnace,and the implied probablity was **28.99 %**
* We found **Nightsaber** has **12.41 %**,winning chnace,and the implied probablity was **6.67 %**
* We found **Welcometomyworld** has **10.53 %**,winning chnace,and the implied probablity was **4.04 %**
* We found **Cookie Crumbs** has **8.65 %**,winning chnace,and the implied probablity was **51.28 %**


As we can see our math model also predict the result somewhat smilar result that were announced [**here**](https://www.equibase.com/premium/eqbPDFChartPlusIndex.cfm?tid=AQU&dt=11/06/2022&ctry=USA) however the best option to bet is **Here We Go Again** as **racebook** thinks that this horse has **3.23 %** winning chance but our math model told that **Here We Go Again** has **16.16 %** chance and we can see he actually **came at 3rd place**. So finding this type of values it will **help us to make more money**. Moreover this are the some main varaibles (Jockey,Trainer, Owner etc) we can add multiple such variables and assign them to a distributed weight percentages. 
