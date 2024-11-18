static int pte_check_hash32(mmu_ctx_t *ctx, target_ulong pte0,

                            target_ulong pte1, int h, int rw, int type)

{

    target_ulong ptem, mmask;

    int access, ret, pteh, ptev, pp;



    ret = -1;

    /* Check validity and table match */

    ptev = pte_is_valid_hash32(pte0);

    pteh = (pte0 >> 6) & 1;

    if (ptev && h == pteh) {

        /* Check vsid & api */

        ptem = pte0 & PTE_PTEM_MASK;

        mmask = PTE_CHECK_MASK;

        pp = pte1 & 0x00000003;

        if (ptem == ctx->ptem) {

            if (ctx->raddr != (hwaddr)-1ULL) {

                /* all matches should have equal RPN, WIMG & PP */

                if ((ctx->raddr & mmask) != (pte1 & mmask)) {

                    qemu_log("Bad RPN/WIMG/PP\n");

                    return -3;

                }

            }

            /* Compute access rights */

            access = pp_check(ctx->key, pp, ctx->nx);

            /* Keep the matching PTE informations */

            ctx->raddr = pte1;

            ctx->prot = access;

            ret = check_prot(ctx->prot, rw, type);

            if (ret == 0) {

                /* Access granted */

                LOG_MMU("PTE access granted !\n");

            } else {

                /* Access right violation */

                LOG_MMU("PTE access rejected\n");

            }

        }

    }



    return ret;

}
