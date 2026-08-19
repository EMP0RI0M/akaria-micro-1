#include "wokwi-api.h"
#include <stdio.h>
#include <stdlib.h>
typedef struct { pin_t distance,temperature,humidity,moisture,light,ir,rx,voltageA,currentA,busA,voltageB,currentB,busB; } chip_state_t;
static pin_t analog_pin(const char *name){ return pin_init(name, ANALOG); }
void chip_init(){
  chip_state_t *s=(chip_state_t*)malloc(sizeof(chip_state_t));
  s->distance=analog_pin("DISTANCE"); s->temperature=analog_pin("TEMPERATURE"); s->humidity=analog_pin("HUMIDITY");
  s->moisture=analog_pin("MOISTURE"); s->light=analog_pin("LIGHT"); s->ir=analog_pin("IR"); s->rx=analog_pin("RX_STATUS");
  s->voltageA=analog_pin("VOLTAGE_A"); s->currentA=analog_pin("CURRENT_A"); s->busA=analog_pin("BUS_A"); s->voltageB=analog_pin("VOLTAGE_B"); s->currentB=analog_pin("CURRENT_B"); s->busB=analog_pin("BUS_B");
  pin_dac_write(s->distance,1.25f); pin_dac_write(s->temperature,1.25f); pin_dac_write(s->humidity,2.5f); pin_dac_write(s->moisture,2.0f); pin_dac_write(s->light,3.0f); pin_dac_write(s->ir,0); pin_dac_write(s->rx,4.5f); pin_dac_write(s->voltageA,4.0f); pin_dac_write(s->currentA,1.0f); pin_dac_write(s->busA,4.0f); pin_dac_write(s->voltageB,4.0f); pin_dac_write(s->currentB,1.0f); pin_dac_write(s->busB,4.0f);
  printf("TESLAX Core safe analog model ready\n");
}
