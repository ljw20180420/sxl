library(ggplot2)
library(tidyverse)
library(ggalluvial)

event_from <- "label1"
event_to <- "label5"

tb <- read_csv("for_LJW/auroc_analysis.csv") |>
    pivot_wider(
        id_cols = c("mouse", "cell"),
        names_from = "label",
        values_from = "type",
    ) |>
    select(mouse, cell, {{event_from}}, {{event_to}})

from_to_count <- tb |>
    summarise(count = n(), .by = c(event_from, event_to))

tb <- tb |>
    left_join(from_to_count, by=c(event_from, event_to))

tb["cross"] <- paste(
    paste(
        tb[[event_from]],
        tb[[event_to]],
        sep="_"
    ),
    tb[["count"]],
    sep=": "
)

tb <- tb |> select(!count)

ggfig <- tb |>
    ggplot(
        aes(axis1 = !!sym(event_from), axis2 = !!sym(event_to), fill=cross)
    ) +
    geom_alluvium() +
    geom_stratum() +
    stat_flow(geom="text", aes(label = cross)) +
    scale_x_discrete(limits = c(event_from, event_to))
    theme_minimal() +
    ggtitle("cell type transition")

ggsave(sprintf("for_LJW/%s_%s.alluvium.pdf", event_from, event_to), ggfig)
