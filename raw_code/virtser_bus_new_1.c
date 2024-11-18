static VirtIOSerialBus *virtser_bus_new(DeviceState *dev)

{

    VirtIOSerialBus *bus;



    bus = FROM_QBUS(VirtIOSerialBus, qbus_create(&virtser_bus_info, dev, NULL));

    bus->qbus.allow_hotplug = 1;



    return bus;

}
