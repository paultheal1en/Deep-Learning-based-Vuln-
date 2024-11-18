static always_inline int get_bat (CPUState *env, mmu_ctx_t *ctx,

                                  target_ulong virtual, int rw, int type)

{

    target_ulong *BATlt, *BATut, *BATu, *BATl;

    target_ulong base, BEPIl, BEPIu, bl;

    int i, pp, pr;

    int ret = -1;



#if defined (DEBUG_BATS)

    if (loglevel != 0) {

        fprintf(logfile, "%s: %cBAT v 0x" ADDRX "\n", __func__,

                type == ACCESS_CODE ? 'I' : 'D', virtual);

    }

#endif

    pr = msr_pr;

    switch (type) {

    case ACCESS_CODE:

        BATlt = env->IBAT[1];

        BATut = env->IBAT[0];

        break;

    default:

        BATlt = env->DBAT[1];

        BATut = env->DBAT[0];

        break;

    }

#if defined (DEBUG_BATS)

    if (loglevel != 0) {

        fprintf(logfile, "%s...: %cBAT v 0x" ADDRX "\n", __func__,

                type == ACCESS_CODE ? 'I' : 'D', virtual);

    }

#endif

    base = virtual & 0xFFFC0000;

    for (i = 0; i < 4; i++) {

        BATu = &BATut[i];

        BATl = &BATlt[i];

        BEPIu = *BATu & 0xF0000000;

        BEPIl = *BATu & 0x0FFE0000;

        bl = (*BATu & 0x00001FFC) << 15;

#if defined (DEBUG_BATS)

        if (loglevel != 0) {

            fprintf(logfile, "%s: %cBAT%d v 0x" ADDRX " BATu 0x" ADDRX

                    " BATl 0x" ADDRX "\n",

                    __func__, type == ACCESS_CODE ? 'I' : 'D', i, virtual,

                    *BATu, *BATl);

        }

#endif

        if ((virtual & 0xF0000000) == BEPIu &&

            ((virtual & 0x0FFE0000) & ~bl) == BEPIl) {

            /* BAT matches */

            if (((pr == 0) && (*BATu & 0x00000002)) ||

                ((pr != 0) && (*BATu & 0x00000001))) {

                /* Get physical address */

                ctx->raddr = (*BATl & 0xF0000000) |

                    ((virtual & 0x0FFE0000 & bl) | (*BATl & 0x0FFE0000)) |

                    (virtual & 0x0001F000);

                /* Compute access rights */

                pp = *BATl & 0x00000003;

                ctx->prot = 0;

                if (pp != 0) {

                    ctx->prot = PAGE_READ | PAGE_EXEC;

                    if (pp == 0x2)

                        ctx->prot |= PAGE_WRITE;

                }

                ret = check_prot(ctx->prot, rw, type);

#if defined (DEBUG_BATS)

                if (ret == 0 && loglevel != 0) {

                    fprintf(logfile, "BAT %d match: r 0x" PADDRX

                            " prot=%c%c\n",

                            i, ctx->raddr, ctx->prot & PAGE_READ ? 'R' : '-',

                            ctx->prot & PAGE_WRITE ? 'W' : '-');

                }

#endif

                break;

            }

        }

    }

    if (ret < 0) {

#if defined (DEBUG_BATS)

        if (loglevel != 0) {

            fprintf(logfile, "no BAT match for 0x" ADDRX ":\n", virtual);

            for (i = 0; i < 4; i++) {

                BATu = &BATut[i];

                BATl = &BATlt[i];

                BEPIu = *BATu & 0xF0000000;

                BEPIl = *BATu & 0x0FFE0000;

                bl = (*BATu & 0x00001FFC) << 15;

                fprintf(logfile, "%s: %cBAT%d v 0x" ADDRX " BATu 0x" ADDRX

                        " BATl 0x" ADDRX " \n\t"

                        "0x" ADDRX " 0x" ADDRX " 0x" ADDRX "\n",

                        __func__, type == ACCESS_CODE ? 'I' : 'D', i, virtual,

                        *BATu, *BATl, BEPIu, BEPIl, bl);

            }

        }

#endif

    }



    /* No hit */

    return ret;

}
