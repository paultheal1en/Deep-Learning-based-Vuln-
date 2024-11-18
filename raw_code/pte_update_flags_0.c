int pte_update_flags(mmu_ctx_t *ctx, target_ulong *pte1p,

                     int ret, int rw)

{

    int store = 0;



    /* Update page flags */

    if (!(*pte1p & 0x00000100)) {

        /* Update accessed flag */

        *pte1p |= 0x00000100;

        store = 1;

    }

    if (!(*pte1p & 0x00000080)) {

        if (rw == 1 && ret == 0) {

            /* Update changed flag */

            *pte1p |= 0x00000080;

            store = 1;

        } else {

            /* Force page fault for first write access */

            ctx->prot &= ~PAGE_WRITE;

        }

    }



    return store;

}
