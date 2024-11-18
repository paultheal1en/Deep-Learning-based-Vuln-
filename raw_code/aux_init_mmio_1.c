void aux_init_mmio(AUXSlave *aux_slave, MemoryRegion *mmio)

{

    assert(!aux_slave->mmio);

    aux_slave->mmio = mmio;

}
