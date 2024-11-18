static int ahci_dma_set_inactive(IDEDMA *dma)

{

    AHCIDevice *ad = DO_UPCAST(AHCIDevice, dma, dma);



    DPRINTF(ad->port_no, "dma done\n");



    /* update d2h status */

    ahci_write_fis_d2h(ad, NULL);



    ad->dma_cb = NULL;



    /* maybe we still have something to process, check later */

    ad->check_bh = qemu_bh_new(ahci_check_cmd_bh, ad);

    qemu_bh_schedule(ad->check_bh);



    return 0;

}
