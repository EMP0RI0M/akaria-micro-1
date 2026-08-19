#if defined(ARDUINO_ARCH_ESP32)
#include <Arduino.h>
void setup(){Serial.begin(115200);Serial2.begin(115200);Serial.println("TESLAX ESP32 GATEWAY READY");}
void loop(){while(Serial2.available())Serial.write(Serial2.read());delay(10);}
#endif
