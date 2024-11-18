I2CBus *aux_get_i2c_bus(AUXBus *bus)

{

    return aux_bridge_get_i2c_bus(bus->bridge);

}
