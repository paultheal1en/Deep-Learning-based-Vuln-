static void qxl_check_state(PCIQXLDevice *d)

{

    QXLRam *ram = d->ram;



    assert(SPICE_RING_IS_EMPTY(&ram->cmd_ring));

    assert(SPICE_RING_IS_EMPTY(&ram->cursor_ring));

}
