static void aer915_class_init(ObjectClass *klass, void *data)

{

    DeviceClass *dc = DEVICE_CLASS(klass);

    I2CSlaveClass *k = I2C_SLAVE_CLASS(klass);



    k->init = aer915_init;

    k->event = aer915_event;

    k->recv = aer915_recv;

    k->send = aer915_send;

    dc->vmsd = &vmstate_aer915_state;

}
