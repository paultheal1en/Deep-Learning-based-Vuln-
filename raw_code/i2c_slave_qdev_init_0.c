static int i2c_slave_qdev_init(DeviceState *dev)

{

    I2CSlave *s = I2C_SLAVE(dev);

    I2CSlaveClass *sc = I2C_SLAVE_GET_CLASS(s);



    return sc->init(s);

}
