static void ide_sector_write(IDEState *s)

{

    int64_t sector_num;

    int ret, n, n1;



    s->status = READY_STAT | SEEK_STAT;

    sector_num = ide_get_sector(s);

#if defined(DEBUG_IDE)

    printf("write sector=%Ld\n", sector_num);

#endif

    n = s->nsector;

    if (n > s->req_nb_sectors)

        n = s->req_nb_sectors;

    ret = bdrv_write(s->bs, sector_num, s->io_buffer, n);

    s->nsector -= n;

    if (s->nsector == 0) {

        /* no more sector to write */

        ide_transfer_stop(s);

    } else {

        n1 = s->nsector;

        if (n1 > s->req_nb_sectors)

            n1 = s->req_nb_sectors;

        ide_transfer_start(s, s->io_buffer, 512 * n1, ide_sector_write);

    }

    ide_set_sector(s, sector_num + n);

    

#ifdef TARGET_I386

    if (win2k_install_hack) {

        /* It seems there is a bug in the Windows 2000 installer HDD

           IDE driver which fills the disk with empty logs when the

           IDE write IRQ comes too early. This hack tries to correct

           that at the expense of slower write performances. Use this

           option _only_ to install Windows 2000. You must disable it

           for normal use. */

        qemu_mod_timer(s->sector_write_timer, 

                       qemu_get_clock(vm_clock) + (ticks_per_sec / 1000));

    } else 

#endif

    {

        ide_set_irq(s);

    }

}
