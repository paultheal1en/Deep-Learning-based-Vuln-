sPAPRTCETable *spapr_tce_find_by_liobn(uint32_t liobn)

{

    sPAPRTCETable *tcet;



    if (liobn & 0xFFFFFFFF00000000ULL) {

        hcall_dprintf("Request for out-of-bounds LIOBN 0x" TARGET_FMT_lx "\n",

                      liobn);

        return NULL;

    }



    QLIST_FOREACH(tcet, &spapr_tce_tables, list) {

        if (tcet->liobn == liobn) {

            return tcet;

        }

    }



    return NULL;

}
