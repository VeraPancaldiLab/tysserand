#install.packages('flowCore')
BiocManager::install("flowCore")
library("flowCore")
library("readr")

data = readr::read_csv(file = '15T011146_80032.fcs.csv')

data$x <- (data$XMin + data$XMin)/2
data$y <- (data$YMin + data$YMin)/2
data$ClassFactor <- paste0(data$`PanCK-CD3+CD8+` , data$`PanCK-CD3+CD8-` , data$`PanCK-CD3+CD20+`, data$`PanCK-CD3-CD20+`, data$`PanCK+CD3-CD8-CD20-`)
data$Class <- as.numeric(as.factor(data$ClassFactor))
fcs