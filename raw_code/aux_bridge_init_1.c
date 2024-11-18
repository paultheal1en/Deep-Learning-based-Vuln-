static void aux_bridge_init(Object *obj)

{

    AUXTOI2CState *s = AUXTOI2C(obj);



    s->i2c_bus = i2c_init_bus(DEVICE(obj), "aux-i2c");

}
