library(ggplot2)
library(tidyverse)
library(ggalluvial)

tb <- read_csv("for_LJW/auroc_analysis.csv")

tb |>
    pivot_wider(
        id_cols = c("mouse", "cell"),
        names_from = "event",
        values_from = "type",
    ) |> 
    ggplot(
        aes(axis1 = event1, axis2 = event5)
    ) +
    scale_x_discrete(limits = c("event1", "event5")) +
    xlab("Demographic") +
    geom_alluvium(aes(fill = event1)) +
    geom_stratum() +
    geom_text(stat = "stratum", aes(label = after_stat(stratum))) +
    theme_minimal() +
    ggtitle("passengers on the maiden voyage of the Titanic",
          "stratified by demographics and survival")
