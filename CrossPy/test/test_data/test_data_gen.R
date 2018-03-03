set.seed(123)
library(tidyverse)

## When importing this data, use the following code
# X_test <- data[,-4]
# y_test <- data[,4]

data_short <- data.frame(X1 = rnorm(10),
                   X2 = rnorm(10),
                   X3 = rnorm(10),
                   X4 = rnorm(10))
data_short <- data_short %>%
  mutate(y = X1 + X2 + X3 + X4) %>%
  select(-X4)


write_csv(data_short, "tests/test_data/test_data_short.csv")

data_long <- data.frame(X1 = rnorm(100),
                   X2 = rnorm(100),
                   X3 = rnorm(100),
                   X4 = rnorm(100))
data_long <- data_long %>%
  mutate(y = X1 + X2 + X3 + X4) %>%
  select(-X4)

write_csv(data_long, "tests/test_data/test_data_short.csv")

