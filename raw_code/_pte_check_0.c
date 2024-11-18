static always_inline int _pte_check (mmu_ctx_t *ctx, int is_64b,

                                     target_ulong pte0, target_ulong pte1,

                                     int h, int rw)

{

    target_ulong ptem, mmask;

    int access, ret, pteh, ptev;



    access = 0;

    ret = -1;

    /* Check validity and table match */

#if defined(TARGET_PPC64)

    if (is_64b) {

        ptev = pte64_is_valid(pte0);

        pteh = (pte0 >> 1) & 1;

    } else

#endif

    {

        ptev = pte_is_valid(pte0);

        pteh = (pte0 >> 6) & 1;

    }

    if (ptev && h == pteh) {

        /* Check vsid & api */

#if defined(TARGET_PPC64)

        if (is_64b) {

            ptem = pte0 & PTE64_PTEM_MASK;

            mmask = PTE64_CHECK_MASK;

        } else

#endif

        {

            ptem = pte0 & PTE_PTEM_MASK;

            mmask = PTE_CHECK_MASK;

        }

        if (ptem == ctx->ptem) {

            if (ctx->raddr != (target_ulong)-1) {

                /* all matches should have equal RPN, WIMG & PP */

                if ((ctx->raddr & mmask) != (pte1 & mmask)) {

                    if (loglevel != 0)

                        fprintf(logfile, "Bad RPN/WIMG/PP\n");

                    return -3;

                }

            }

            /* Compute access rights */

            if (ctx->key == 0) {

                access = PAGE_READ;

                if ((pte1 & 0x00000003) != 0x3)

                    access |= PAGE_WRITE;

            } else {

                switch (pte1 & 0x00000003) {

                case 0x0:

                    access = 0;

                    break;

                case 0x1:

                case 0x3:

                    access = PAGE_READ;

                    break;

                case 0x2:

                    access = PAGE_READ | PAGE_WRITE;

                    break;

                }

            }

            /* Keep the matching PTE informations */

            ctx->raddr = pte1;

            ctx->prot = access;

            if ((rw == 0 && (access & PAGE_READ)) ||

                (rw == 1 && (access & PAGE_WRITE))) {

                /* Access granted */

#if defined (DEBUG_MMU)

                if (loglevel != 0)

                    fprintf(logfile, "PTE access granted !\n");

#endif

                ret = 0;

            } else {

                /* Access right violation */

#if defined (DEBUG_MMU)

                if (loglevel != 0)

                    fprintf(logfile, "PTE access rejected\n");

#endif

                ret = -2;

            }

        }

    }



    return ret;

}
