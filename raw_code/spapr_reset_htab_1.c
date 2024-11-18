static void spapr_reset_htab(sPAPRMachineState *spapr)

{

    long shift;

    int index;



    shift = kvmppc_reset_htab(spapr->htab_shift);

    if (shift > 0) {

        if (shift != spapr->htab_shift) {

            error_setg(&error_abort, "Requested HTAB allocation failed during reset");

        }



        /* Tell readers to update their file descriptor */

        if (spapr->htab_fd >= 0) {

            spapr->htab_fd_stale = true;

        }

    } else {

        memset(spapr->htab, 0, HTAB_SIZE(spapr));



        for (index = 0; index < HTAB_SIZE(spapr) / HASH_PTE_SIZE_64; index++) {

            DIRTY_HPTE(HPTE(spapr->htab, index));

        }

    }



    /* Update the RMA size if necessary */

    if (spapr->vrma_adjust) {

        spapr->rma_size = kvmppc_rma_size(spapr_node0_size(),

                                          spapr->htab_shift);

    }

}
