static void continue_after_map_failure(void *opaque)

{

    DMAAIOCB *dbs = (DMAAIOCB *)opaque;



    dbs->bh = qemu_bh_new(reschedule_dma, dbs);

    qemu_bh_schedule(dbs->bh);

}
