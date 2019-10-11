import pandas as pd

# CSV Category set
rd = pd.read_csv("6_rawdata.csv")
rd.columns = ["Bottom", "Left", "Right", "Heading", "Top", "angle", "reallatralspeed", "lanemarker"]
#Bottom, Heading
rd.to_csv("dataset10.csv",header=True,index=False)

