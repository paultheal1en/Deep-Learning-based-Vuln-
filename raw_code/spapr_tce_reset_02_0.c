void spapr_tce_reset(DMAContext *dma)

{

    if (dma) {

        sPAPRTCETable *tcet = DO_UPCAST(sPAPRTCETable, dma, dma);

        size_t table_size = (tcet->window_size >> SPAPR_TCE_PAGE_SHIFT)

            * sizeof(sPAPRTCE);



        memset(tcet->table, 0, table_size);

    }

}
