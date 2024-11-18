static int get_segment (CPUState *env, mmu_ctx_t *ctx,

                        target_ulong eaddr, int rw, int type)

{

    target_phys_addr_t sdr, hash, mask, sdr_mask;

    target_ulong sr, vsid, vsid_mask, pgidx, page_mask;

#if defined(TARGET_PPC64)

    int attr;

#endif

    int ds, nx, vsid_sh, sdr_sh;

    int ret, ret2;



#if defined(TARGET_PPC64)

    if (env->mmu_model == POWERPC_MMU_64B ||

        env->mmu_model == POWERPC_MMU_64BRIDGE) {

        ret = slb_lookup(env, eaddr, &vsid, &page_mask, &attr);

        if (ret < 0)

            return ret;

        ctx->key = ((attr & 0x40) && msr_pr == 1) ||

            ((attr & 0x80) && msr_pr == 0) ? 1 : 0;

        ds = 0;

        nx = attr & 0x20 ? 1 : 0;

        vsid_mask = 0x00003FFFFFFFFF80ULL;

        vsid_sh = 7;

        sdr_sh = 18;

        sdr_mask = 0x3FF80;

    } else

#endif /* defined(TARGET_PPC64) */

    {

        sr = env->sr[eaddr >> 28];

        page_mask = 0x0FFFFFFF;

        ctx->key = (((sr & 0x20000000) && msr_pr == 1) ||

                    ((sr & 0x40000000) && msr_pr == 0)) ? 1 : 0;

        ds = sr & 0x80000000 ? 1 : 0;

        nx = sr & 0x10000000 ? 1 : 0;

        vsid = sr & 0x00FFFFFF;

        vsid_mask = 0x01FFFFC0;

        vsid_sh = 6;

        sdr_sh = 16;

        sdr_mask = 0xFFC0;

#if defined (DEBUG_MMU)

        if (loglevel != 0) {

            fprintf(logfile, "Check segment v=0x" ADDRX " %d 0x" ADDRX

                    " nip=0x" ADDRX " lr=0x" ADDRX

                    " ir=%d dr=%d pr=%d %d t=%d\n",

                    eaddr, (int)(eaddr >> 28), sr, env->nip,

                    env->lr, msr_ir, msr_dr, msr_pr, rw, type);

        }

        if (!ds && loglevel != 0) {

            fprintf(logfile, "pte segment: key=%d n=0x" ADDRX "\n",

                    ctx->key, sr & 0x10000000);

        }

#endif

    }

    ret = -1;

    if (!ds) {

        /* Check if instruction fetch is allowed, if needed */

        if (type != ACCESS_CODE || nx == 0) {

            /* Page address translation */

            pgidx = (eaddr & page_mask) >> TARGET_PAGE_BITS;

            hash = ((vsid ^ pgidx) << vsid_sh) & vsid_mask;

            /* Primary table address */

            sdr = env->sdr1;

            mask = ((sdr & 0x000001FF) << sdr_sh) | sdr_mask;

            ctx->pg_addr[0] = get_pgaddr(sdr, sdr_sh, hash, mask);

            /* Secondary table address */

            hash = (~hash) & vsid_mask;

            ctx->pg_addr[1] = get_pgaddr(sdr, sdr_sh, hash, mask);

#if defined(TARGET_PPC64)

            if (env->mmu_model == POWERPC_MMU_64B ||

                env->mmu_model == POWERPC_MMU_64BRIDGE) {

                /* Only 5 bits of the page index are used in the AVPN */

                ctx->ptem = (vsid << 12) | ((pgidx >> 4) & 0x0F80);

            } else

#endif

            {

                ctx->ptem = (vsid << 7) | (pgidx >> 10);

            }

            /* Initialize real address with an invalid value */

            ctx->raddr = (target_ulong)-1;

            if (unlikely(env->mmu_model == POWERPC_MMU_SOFT_6xx ||

                         env->mmu_model == POWERPC_MMU_SOFT_74xx)) {

                /* Software TLB search */

                ret = ppc6xx_tlb_check(env, ctx, eaddr, rw, type);

            } else {

#if defined (DEBUG_MMU)

                if (loglevel != 0) {

                    fprintf(logfile, "0 sdr1=0x" PADDRX " vsid=0x%06x "

                            "api=0x%04x hash=0x%07x pg_addr=0x" PADDRX "\n",

                            sdr, (uint32_t)vsid, (uint32_t)pgidx,

                            (uint32_t)hash, ctx->pg_addr[0]);

                }

#endif

                /* Primary table lookup */

                ret = find_pte(env, ctx, 0, rw);

                if (ret < 0) {

                    /* Secondary table lookup */

#if defined (DEBUG_MMU)

                    if (eaddr != 0xEFFFFFFF && loglevel != 0) {

                        fprintf(logfile,

                                "1 sdr1=0x" PADDRX " vsid=0x%06x api=0x%04x "

                                "hash=0x%05x pg_addr=0x" PADDRX "\n",

                                sdr, (uint32_t)vsid, (uint32_t)pgidx,

                                (uint32_t)hash, ctx->pg_addr[1]);

                    }

#endif

                    ret2 = find_pte(env, ctx, 1, rw);

                    if (ret2 != -1)

                        ret = ret2;

                }

            }

        } else {

#if defined (DEBUG_MMU)

            if (loglevel != 0)

                fprintf(logfile, "No access allowed\n");

#endif

            ret = -3;

        }

    } else {

#if defined (DEBUG_MMU)

        if (loglevel != 0)

            fprintf(logfile, "direct store...\n");

#endif

        /* Direct-store segment : absolutely *BUGGY* for now */

        switch (type) {

        case ACCESS_INT:

            /* Integer load/store : only access allowed */

            break;

        case ACCESS_CODE:

            /* No code fetch is allowed in direct-store areas */

            return -4;

        case ACCESS_FLOAT:

            /* Floating point load/store */

            return -4;

        case ACCESS_RES:

            /* lwarx, ldarx or srwcx. */

            return -4;

        case ACCESS_CACHE:

            /* dcba, dcbt, dcbtst, dcbf, dcbi, dcbst, dcbz, or icbi */

            /* Should make the instruction do no-op.

             * As it already do no-op, it's quite easy :-)

             */

            ctx->raddr = eaddr;

            return 0;

        case ACCESS_EXT:

            /* eciwx or ecowx */

            return -4;

        default:

            if (logfile) {

                fprintf(logfile, "ERROR: instruction should not need "

                        "address translation\n");

            }

            return -4;

        }

        if ((rw == 1 || ctx->key != 1) && (rw == 0 || ctx->key != 0)) {

            ctx->raddr = eaddr;

            ret = 2;

        } else {

            ret = -2;

        }

    }



    return ret;

}
