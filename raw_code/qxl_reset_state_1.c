static void qxl_reset_state(PCIQXLDevice *d)

{

    QXLRam *ram = d->ram;

    QXLRom *rom = d->rom;



    assert(SPICE_RING_IS_EMPTY(&ram->cmd_ring));

    assert(SPICE_RING_IS_EMPTY(&ram->cursor_ring));

    d->shadow_rom.update_id = cpu_to_le32(0);

    *rom = d->shadow_rom;

    qxl_rom_set_dirty(d);

    init_qxl_ram(d);

    d->num_free_res = 0;

    d->last_release = NULL;

    memset(&d->ssd.dirty, 0, sizeof(d->ssd.dirty));

}
