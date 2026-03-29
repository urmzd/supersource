#!/usr/bin/env Rscript
s_1 <- 30659.46
s_2 <- 53453.19
s_3 <- 221602.52
s_4 <- 195000.99

se <- function(s_1, s_2, s_3 = 0, s_4 = 0, s_len = 56) {
  mse <- sqrt((s_1 + s_3 + s_2 + s_4) / (s_len * 4 - 4))

  s_e <- mse * sqrt(2/(s_len))
  s_e
}

print(se(s_1, s_2, s_3, s_4))

u_1 = 23.27
u_2 = 20.00
u_3 = 21.52
u_4 = 50.19


ans = (u_1 - u_4)/se(s_1, s_2, s_3, s_4)
ans

t_star <- (0.05 / 3) / 2
t_val <- qt(1 - t_star, (56 * 4 - 4))
t_val

left = (u_1 - u_3) - (t_val * se(s_1, s_2, s_3, s_4))
right = (u_1 - u_3) + (t_val * se(s_1, s_2, s_3, s_4))

left
right



