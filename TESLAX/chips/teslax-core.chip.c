#include "wokwi-api.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
static float clampf(float x,float lo,float hi);
typedef struct { pin_t distance,temperature,humidity,moisture,light,ir,rx,voltageA,currentA,busA,voltageB,currentB,busB,faultAOut,faultBOut,statusOut; uint32_t a,b,t,busAAttr,busBAttr,faultA,faultB; timer_t timer; } chip_state_t;
static pin_t ap(const char *n){ return pin_init(n, ANALOG); }
static void refresh(void *data){chip_state_t *s=(chip_state_t*)data;float distance=clampf(attr_read_float(s->b),10,300),alignment=clampf(attr_read_float(s->a)/100,0,1),tx=clampf(attr_read_float(s->t)/100,0,1);float efficiency=clampf(alignment/(1.0f+distance/100.0f),0,1),received=clampf(tx*efficiency,0,1),busA=clampf(attr_read_float(s->busAAttr),0,5),busB=clampf(attr_read_float(s->busBAttr)*received,0,5);pin_dac_write(s->distance,distance/60.0f);pin_dac_write(s->rx,received>0.05f?4.5f:0);pin_dac_write(s->voltageA,busA);pin_dac_write(s->currentA,tx*2.5f);pin_dac_write(s->busA,busA);pin_dac_write(s->voltageB,busB);pin_dac_write(s->currentB,received*2.5f);pin_dac_write(s->busB,busB);pin_write(s->faultAOut,attr_read(s->faultA)?HIGH:LOW);pin_write(s->faultBOut,attr_read(s->faultB)?HIGH:LOW);pin_write(s->statusOut,received>0.05f?HIGH:LOW);}
static float clampf(float x,float lo,float hi){return x<lo?lo:(x>hi?hi:x);}
void chip_init(){
 chip_state_t *s=(chip_state_t*)calloc(1,sizeof(chip_state_t));
 s->distance=ap("DISTANCE");s->temperature=ap("TEMPERATURE");s->humidity=ap("HUMIDITY");s->moisture=ap("MOISTURE");s->light=ap("LIGHT");s->ir=ap("IR");s->rx=ap("RX_STATUS");s->voltageA=ap("VOLTAGE_A");s->currentA=ap("CURRENT_A");s->busA=ap("BUS_A");s->voltageB=ap("VOLTAGE_B");s->currentB=ap("CURRENT_B");s->busB=ap("BUS_B");s->faultAOut=pin_init("FAULT_A",OUTPUT_LOW);s->faultBOut=pin_init("FAULT_B",OUTPUT_LOW);s->statusOut=pin_init("STATUS",OUTPUT_LOW);
 s->a=attr_init_float("alignment",80);s->t=attr_init_float("transmitLevel",80);s->b=attr_init_float("distance",40);s->busAAttr=attr_init_float("busVoltageA",4);s->busBAttr=attr_init_float("busVoltageB",4);s->faultA=attr_init("coilFaultA",0);s->faultB=attr_init("coilFaultB",0);
 float distance=clampf(attr_read_float(s->b),10,300), alignment=clampf(attr_read_float(s->a)/100,0,1), tx=clampf(attr_read_float(s->t)/100,0,1);float efficiency=clampf(alignment*(1.0f/(1.0f+distance/100.0f)),0,1);float received=clampf(tx*efficiency,0,1);float busA=clampf(attr_read_float(s->busAAttr),0,5),busB=clampf(attr_read_float(s->busBAttr)*received,0,5);
 pin_dac_write(s->distance,clampf(distance/300*5,0,5));pin_dac_write(s->temperature,1.25f);pin_dac_write(s->humidity,2.5f);pin_dac_write(s->moisture,2);pin_dac_write(s->light,3);pin_dac_write(s->ir,0);pin_dac_write(s->rx,(received>0.05f)?4.5f:0);pin_dac_write(s->voltageA,busA);pin_dac_write(s->currentA,clampf(tx,0,1)*2.5f);pin_dac_write(s->busA,busA);pin_dac_write(s->voltageB,busB);pin_dac_write(s->currentB,clampf(received,0,1)*2.5f);pin_dac_write(s->busB,busB);
 timer_config_t tc={.callback=refresh,.user_data=s};s->timer=timer_init(&tc);timer_start(s->timer,100000,true);refresh(s);printf("TESLAX Core model distance=%.1f alignment=%.2f tx=%.2f efficiency=%.2f received=%.2f\n",distance,alignment,tx,efficiency,received);
}
