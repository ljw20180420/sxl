library(ggplot2)
library(tidyverse)
library(ggalluvial)

event_from <- "label1"
event_to <- "label5"
# mouses <- c("F2_2", "F2_3")
# mouses <- c("F2_2")

tb <- read_csv("for_LJW/auroc_analysis.csv") |>
    # filter(mouse %in% mouses) |>
    pivot_wider(
        id_cols = c("mouse", "cell"),
        names_from = "label",
        values_from = "type",
    )

from_count <- tb  |>
    summarise(count = n(), .by = c(event_from))

from_to_count <- tb |>
    summarise(count = n(), .by = c(event_from, event_to))

tb <- tb |>
    left_join(from_count, by=c(event_from)) |>
    mutate(from = sprintf("%s: %d", event_from, count)) |>
    select(!count)

tb <- tb |>
    left_join(from_to_count, by=c(event_from, event_to)) |>
    mutate(from_to = sprintf("%s_%s: %d", event_from, event_to, count)) |>
    select(!count)


ggfig <- tb |>
    ggplot(
        aes(axis1 = from, axis2 = from_to)
    ) +
    scale_x_discrete(limits = c(event_from, event_to)) +
    xlab("transit") +
    ylab("count") +
    geom_alluvium(aes(fill = from)) +
    geom_stratum() +
    geom_text(stat = "stratum", aes(label = after_stat(stratum))) +
    theme_minimal() +
    ggtitle("cell type transition")

ggsave(sprintf("for_LJW/%s_%s.alluvium.pdf", event_from, event_to), ggfig)
