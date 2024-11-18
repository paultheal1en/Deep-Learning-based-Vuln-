void spapr_tce_reset(sPAPRTCETable *tcet)

{

    size_t table_size = (tcet->window_size >> SPAPR_TCE_PAGE_SHIFT)

        * sizeof(sPAPRTCE);



    tcet->bypass = false;

    memset(tcet->table, 0, table_size);

}
