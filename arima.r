# r-studio mei run karna saare commands

install.packages("forecast")
library(forecast)

data("AirPassengers")
passengers_ts <- ts(AirPassengers, frequency=12)
plot(passengers_ts, main=AirPassengers)
arima_model <- auto.arima(passengers_ts)
summary(arima_model)

forecast_plot <- forecast(arima_model, h=24)
plot(forecast_plot, main="Forecast for airline passengers")