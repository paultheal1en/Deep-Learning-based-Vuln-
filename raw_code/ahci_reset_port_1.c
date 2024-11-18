static void ahci_reset_port(AHCIState *s, int port)
{
    AHCIDevice *d = &s->dev[port];
    AHCIPortRegs *pr = &d->port_regs;
    IDEState *ide_state = &d->port.ifs[0];
    int i;
    DPRINTF(port, "reset port\n");
    ide_bus_reset(&d->port);
    ide_state->ncq_queues = AHCI_MAX_CMDS;
    pr->scr_stat = 0;
    pr->scr_err = 0;
    pr->scr_act = 0;
    d->busy_slot = -1;
    d->init_d2h_sent = 0;
    ide_state = &s->dev[port].port.ifs[0];
    if (!ide_state->bs) {
        return;
    /* reset ncq queue */
    for (i = 0; i < AHCI_MAX_CMDS; i++) {
        NCQTransferState *ncq_tfs = &s->dev[port].ncq_tfs[i];
        if (ncq_tfs->aiocb) {
            bdrv_aio_cancel(ncq_tfs->aiocb);
            ncq_tfs->aiocb = NULL;
        qemu_sglist_destroy(&ncq_tfs->sglist);
        ncq_tfs->used = 0;
    s->dev[port].port_state = STATE_RUN;
    if (!ide_state->bs) {
        s->dev[port].port_regs.sig = 0;
        ide_state->status = SEEK_STAT | WRERR_STAT;
    } else if (ide_state->drive_kind == IDE_CD) {
        s->dev[port].port_regs.sig = SATA_SIGNATURE_CDROM;
        ide_state->lcyl = 0x14;
        ide_state->hcyl = 0xeb;
        DPRINTF(port, "set lcyl = %d\n", ide_state->lcyl);
        ide_state->status = SEEK_STAT | WRERR_STAT | READY_STAT;
    } else {
        s->dev[port].port_regs.sig = SATA_SIGNATURE_DISK;
        ide_state->status = SEEK_STAT | WRERR_STAT;
    ide_state->error = 1;
    ahci_init_d2h(d);