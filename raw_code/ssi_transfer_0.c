uint32_t ssi_transfer(SSIBus *bus, uint32_t val)

{

    DeviceState *dev;

    SSISlave *slave;

    dev = LIST_FIRST(&bus->qbus.children);

    if (!dev) {

        return 0;

    }

    slave = SSI_SLAVE_FROM_QDEV(dev);

    return slave->info->transfer(slave, val);

}
