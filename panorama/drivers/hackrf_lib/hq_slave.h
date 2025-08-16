// hq_slave.h
#ifndef HQ_SLAVE_H
#define HQ_SLAVE_H

#include "hq_init.h"

int start_slave(SdrCtx* slave, int slave_idx);
void stop_slave(SdrCtx* slave);

#endif // HQ_SLAVE_H