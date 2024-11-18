void i2c_register_slave(I2CSlaveInfo *info)

{

    assert(info->qdev.size >= sizeof(i2c_slave));

    info->qdev.init = i2c_slave_qdev_init;

    info->qdev.bus_type = BUS_TYPE_I2C;

    qdev_register(&info->qdev);

}
