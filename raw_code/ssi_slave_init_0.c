static int ssi_slave_init(DeviceState *dev, DeviceInfo *base_info)

{

    SSISlaveInfo *info = container_of(base_info, SSISlaveInfo, qdev);

    SSISlave *s = SSI_SLAVE_FROM_QDEV(dev);

    SSIBus *bus;



    bus = FROM_QBUS(SSIBus, qdev_get_parent_bus(dev));

    if (LIST_FIRST(&bus->qbus.children) != dev

        || LIST_NEXT(dev, sibling) != NULL) {

        hw_error("Too many devices on SSI bus");

    }



    s->info = info;

    return info->init(s);

}
