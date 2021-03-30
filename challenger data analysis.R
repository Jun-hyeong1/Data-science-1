chall <- read.csv('https://raw.githubusercontent.com/stedy/Machine-learning-with-R-datasets/master/challenger.csv')

install.packages("dplyr")
install.packages("ggplot2")

library(dplyr)
library(ggplot2)

chall <- tbl_df(chall)
glimpse(chall)
