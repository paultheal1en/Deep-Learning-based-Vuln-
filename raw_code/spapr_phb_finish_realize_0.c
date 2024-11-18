static void spapr_phb_finish_realize(sPAPRPHBState *sphb, Error **errp)

{

    sphb->dma_window_start = 0;

    sphb->dma_window_size = 0x40000000;

    sphb->tcet = spapr_tce_new_table(DEVICE(sphb), sphb->dma_liobn,

                                     sphb->dma_window_size);

    if (!sphb->tcet) {

        error_setg(errp, "Unable to create TCE table for %s",

                   sphb->dtbusname);

        return ;

    }

    address_space_init(&sphb->iommu_as, spapr_tce_get_iommu(sphb->tcet),

                       sphb->dtbusname);

}
