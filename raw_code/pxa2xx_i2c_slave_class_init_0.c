static void pxa2xx_i2c_slave_class_init(ObjectClass *klass, void *data)

{

    I2CSlaveClass *k = I2C_SLAVE_CLASS(klass);



    k->init = pxa2xx_i2c_slave_init;

    k->event = pxa2xx_i2c_event;

    k->recv = pxa2xx_i2c_rx;

    k->send = pxa2xx_i2c_tx;

}
