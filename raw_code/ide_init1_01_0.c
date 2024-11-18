static void ide_init1(IDEBus *bus, int unit, DriveInfo *dinfo)

{

    static int drive_serial = 1;

    IDEState *s = &bus->ifs[unit];



    s->bus = bus;

    s->unit = unit;

    s->drive_serial = drive_serial++;

    s->io_buffer = qemu_blockalign(s->bs, IDE_DMA_BUF_SECTORS*512 + 4);

    s->io_buffer_total_len = IDE_DMA_BUF_SECTORS*512 + 4;

    s->smart_selftest_data = qemu_blockalign(s->bs, 512);

    s->sector_write_timer = qemu_new_timer(vm_clock,

                                           ide_sector_write_timer_cb, s);

    ide_init_drive(s, dinfo, NULL);

}
