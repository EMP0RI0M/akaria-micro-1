# TESLAX Wokwi Demonstrator

Safe low-voltage simulation of protected wireless-energy transfer. The Mega is the deterministic controller. Coil coupling, rectifiers, filters, DC buses, microgrids, moisture/IR/RX telemetry, Raspberry Pi 4, and micro:bit are labeled safe model/interface blocks; no high-voltage construction is modeled.

## Pin map
| Function | Mega pin |
|---|---:|
| HC-SR04 TRIG/ECHO | 22 / 23 |
| DHT22 DATA | 24 |
| IR, moisture, light, RX status | 25, A0, A1, A2 |
| Bus A/B telemetry | A3 / A4 |
| Buzzer | 6 |
| Relay A/B | 40 / 41 |
| Coil A/B models | 42 / 43 |
| Rectifier A/B models | 44 / 45 |
| Status, fault A, fault B, system LEDs | 46 / 47 / 48 / 49 |
| LCD RS,E,D4-D7 | 30,31,32,33,34,35 |
| ESP32 bridge UART | TX1=18, RX1=19 |

Directly simulated: Mega, ESP32 DevKit, HC-SR04, DHT22, photoresistor, LCD, LEDs, buzzer, relay, and passive resistors. L298N/motors are optional auxiliary loads and are not powered from Mega 5V.
