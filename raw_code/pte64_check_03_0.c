static int pte64_check(mmu_ctx_t *ctx, target_ulong pte0,

                       target_ulong pte1, int h, int rw, int type)

{

    target_ulong ptem, mmask;

    int access, ret, pteh, ptev, pp;



    ret = -1;

    /* Check validity and table match */

    ptev = pte64_is_valid(pte0);

    pteh = (pte0 >> 1) & 1;

    if (ptev && h == pteh) {

        /* Check vsid & api */

        ptem = pte0 & PTE64_PTEM_MASK;

        mmask = PTE64_CHECK_MASK;

        pp = (pte1 & 0x00000003) | ((pte1 >> 61) & 0x00000004);

        ctx->nx  = (pte1 >> 2) & 1; /* No execute bit */

        ctx->nx |= (pte1 >> 3) & 1; /* Guarded bit    */

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
