SSIBus *ssi_create_bus(DeviceState *parent, const char *name)

{

    BusState *bus;

    bus = qbus_create(BUS_TYPE_SSI, sizeof(SSIBus), parent, name);

    return FROM_QBUS(SSIBus, bus);

}
