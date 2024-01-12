import pandas as pd

# creating and initializing a list
values= [['Rohan',455],['Elvish',250],['Deepak',495],
         ['Soni',400],['Radhika',350],['Vansh',450]]
print("Values:")
print(values)
print("\n")

df = pd.DataFrame(values,columns=['Name','Total_Marks'])

# apply lambda function to find percentage
df = df.assign(Product=lambda x: (x['Total_Marks']/500 * 100))
print(df)

