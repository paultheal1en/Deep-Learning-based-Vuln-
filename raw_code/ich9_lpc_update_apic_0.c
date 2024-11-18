static void ich9_lpc_update_apic(ICH9LPCState *lpc, int gsi)

{

    int level = 0;



    assert(gsi >= ICH9_LPC_PIC_NUM_PINS);



    level |= pci_bus_get_irq_level(lpc->d.bus, ich9_gsi_to_pirq(gsi));

    if (gsi == lpc->sci_gsi) {

        level |= lpc->sci_level;

    }



    qemu_set_irq(lpc->gsi[gsi], level);

}
