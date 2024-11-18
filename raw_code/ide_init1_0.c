static void ide_init1(IDEBus *bus, int unit)

{

    static int drive_serial = 1;

    IDEState *s = &bus->ifs[unit];



    s->bus = bus;

    s->unit = unit;

    s->drive_serial = drive_serial++;

    /* we need at least 2k alignment for accessing CDROMs using O_DIRECT */

    s->io_buffer_total_len = IDE_DMA_BUF_SECTORS*512 + 4;

    s->io_buffer = qemu_memalign(2048, s->io_buffer_total_len);

    memset(s->io_buffer, 0, s->io_buffer_total_len);



    s->smart_selftest_data = qemu_blockalign(s->bs, 512);

    memset(s->smart_selftest_data, 0, 512);



    s->sector_write_timer = timer_new_ns(QEMU_CLOCK_VIRTUAL,

                                           ide_sector_write_timer_cb, s);

}
