static void ahci_write_fis_sdb(AHCIState *s, int port, uint32_t finished)

{

    AHCIPortRegs *pr = &s->dev[port].port_regs;

    IDEState *ide_state;

    uint8_t *sdb_fis;



    if (!s->dev[port].res_fis ||

        !(pr->cmd & PORT_CMD_FIS_RX)) {

        return;

    }



    sdb_fis = &s->dev[port].res_fis[RES_FIS_SDBFIS];

    ide_state = &s->dev[port].port.ifs[0];



    /* clear memory */

    *(uint32_t*)sdb_fis = 0;



    /* write values */

    sdb_fis[0] = ide_state->error;

    sdb_fis[2] = ide_state->status & 0x77;

    s->dev[port].finished |= finished;

    *(uint32_t*)(sdb_fis + 4) = cpu_to_le32(s->dev[port].finished);



    ahci_trigger_irq(s, &s->dev[port], PORT_IRQ_SDB_FIS);

}
