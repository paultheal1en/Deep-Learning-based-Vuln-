int mmu40x_get_physical_address (CPUState *env, mmu_ctx_t *ctx,

                                 target_ulong address, int rw, int access_type)

{

    ppcemb_tlb_t *tlb;

    target_phys_addr_t raddr;

    int i, ret, zsel, zpr;



    ret = -1;

    raddr = -1;

    for (i = 0; i < env->nb_tlb; i++) {

        tlb = &env->tlb[i].tlbe;

        if (ppcemb_tlb_check(env, tlb, &raddr, address,

                             env->spr[SPR_40x_PID], 0, i) < 0)

            continue;

        zsel = (tlb->attr >> 4) & 0xF;

        zpr = (env->spr[SPR_40x_ZPR] >> (28 - (2 * zsel))) & 0x3;

#if defined (DEBUG_SOFTWARE_TLB)

        if (loglevel != 0) {

            fprintf(logfile, "%s: TLB %d zsel %d zpr %d rw %d attr %08x\n",

                    __func__, i, zsel, zpr, rw, tlb->attr);

        }

#endif

        if (access_type == ACCESS_CODE) {

            /* Check execute enable bit */

            switch (zpr) {

            case 0x2:

                if (msr_pr)

                    goto check_exec_perm;

                goto exec_granted;

            case 0x0:

                if (msr_pr) {

                    ctx->prot = 0;

                    ret = -3;

                    break;

                }

                /* No break here */

            case 0x1:

            check_exec_perm:

                /* Check from TLB entry */

                if (!(tlb->prot & PAGE_EXEC)) {

                    ret = -3;

                } else {

                    if (tlb->prot & PAGE_WRITE) {

                        ctx->prot = PAGE_READ | PAGE_WRITE;

                    } else {

                        ctx->prot = PAGE_READ;

                    }

                    ret = 0;

                }

                break;

            case 0x3:

            exec_granted:

                /* All accesses granted */

                ctx->prot = PAGE_READ | PAGE_WRITE;

                ret = 0;

                break;

            }

        } else {

            switch (zpr) {

            case 0x2:

                if (msr_pr)

                    goto check_rw_perm;

                goto rw_granted;

            case 0x0:

                if (msr_pr) {

                    ctx->prot = 0;

                    ret = -2;

                    break;

                }

                /* No break here */

            case 0x1:

            check_rw_perm:

                /* Check from TLB entry */

                /* Check write protection bit */

                if (tlb->prot & PAGE_WRITE) {

                    ctx->prot = PAGE_READ | PAGE_WRITE;

                    ret = 0;

                } else {

                    ctx->prot = PAGE_READ;

                    if (rw)

                        ret = -2;

                    else

                        ret = 0;

                }

                break;

            case 0x3:

            rw_granted:

                /* All accesses granted */

                ctx->prot = PAGE_READ | PAGE_WRITE;

                ret = 0;

                break;

            }

        }

        if (ret >= 0) {

            ctx->raddr = raddr;

#if defined (DEBUG_SOFTWARE_TLB)

            if (loglevel != 0) {

                fprintf(logfile, "%s: access granted " ADDRX " => " REGX

                        " %d %d\n", __func__, address, ctx->raddr, ctx->prot,

                        ret);

            }

#endif

            return 0;

        }

    }

#if defined (DEBUG_SOFTWARE_TLB)

    if (loglevel != 0) {

        fprintf(logfile, "%s: access refused " ADDRX " => " REGX

                " %d %d\n", __func__, address, raddr, ctx->prot,

                ret);

    }

#endif



    return ret;

}
