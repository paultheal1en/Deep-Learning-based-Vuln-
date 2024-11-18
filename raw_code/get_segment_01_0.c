static inline int get_segment(CPUState *env, mmu_ctx_t *ctx,

                              target_ulong eaddr, int rw, int type)

{

    target_phys_addr_t sdr, hash, mask, sdr_mask, htab_mask;

    target_ulong sr, vsid, vsid_mask, pgidx, page_mask;

    int ds, vsid_sh, sdr_sh, pr, target_page_bits;

    int ret, ret2;



    pr = msr_pr;

#if defined(TARGET_PPC64)

    if (env->mmu_model & POWERPC_MMU_64) {

        ppc_slb_t *slb;



        LOG_MMU("Check SLBs\n");

        slb = slb_lookup(env, eaddr);

        if (!slb) {

            return -5;

        }



        vsid = (slb->vsid & SLB_VSID_VSID) >> SLB_VSID_SHIFT;

        page_mask = ~SEGMENT_MASK_256M;

        target_page_bits = (slb->vsid & SLB_VSID_L)

            ? TARGET_PAGE_BITS_16M : TARGET_PAGE_BITS;

        ctx->key = !!(pr ? (slb->vsid & SLB_VSID_KP)

                      : (slb->vsid & SLB_VSID_KS));

        ds = 0;

        ctx->nx = !!(slb->vsid & SLB_VSID_N);

        ctx->eaddr = eaddr;

        vsid_mask = 0x00003FFFFFFFFF80ULL;

        vsid_sh = 7;

        sdr_sh = 18;

        sdr_mask = 0x3FF80;

    } else

#endif /* defined(TARGET_PPC64) */

    {

        sr = env->sr[eaddr >> 28];

        page_mask = 0x0FFFFFFF;

        ctx->key = (((sr & 0x20000000) && (pr != 0)) ||

                    ((sr & 0x40000000) && (pr == 0))) ? 1 : 0;

        ds = sr & 0x80000000 ? 1 : 0;

        ctx->nx = sr & 0x10000000 ? 1 : 0;

        vsid = sr & 0x00FFFFFF;

        vsid_mask = 0x01FFFFC0;

        vsid_sh = 6;

        sdr_sh = 16;

        sdr_mask = 0xFFC0;

        target_page_bits = TARGET_PAGE_BITS;

        LOG_MMU("Check segment v=" TARGET_FMT_lx " %d " TARGET_FMT_lx " nip="

                TARGET_FMT_lx " lr=" TARGET_FMT_lx

                " ir=%d dr=%d pr=%d %d t=%d\n",

                eaddr, (int)(eaddr >> 28), sr, env->nip, env->lr, (int)msr_ir,

                (int)msr_dr, pr != 0 ? 1 : 0, rw, type);

    }

    LOG_MMU("pte segment: key=%d ds %d nx %d vsid " TARGET_FMT_lx "\n",

            ctx->key, ds, ctx->nx, vsid);

    ret = -1;

    if (!ds) {

        /* Check if instruction fetch is allowed, if needed */

        if (type != ACCESS_CODE || ctx->nx == 0) {

            /* Page address translation */

            /* Primary table address */

            sdr = env->sdr1;

            pgidx = (eaddr & page_mask) >> target_page_bits;

#if defined(TARGET_PPC64)

            if (env->mmu_model & POWERPC_MMU_64) {

                htab_mask = 0x0FFFFFFF >> (28 - (sdr & 0x1F));

                /* XXX: this is false for 1 TB segments */

                hash = ((vsid ^ pgidx) << vsid_sh) & vsid_mask;

            } else

#endif

            {

                htab_mask = sdr & 0x000001FF;

                hash = ((vsid ^ pgidx) << vsid_sh) & vsid_mask;

            }

            mask = (htab_mask << sdr_sh) | sdr_mask;

            LOG_MMU("sdr " TARGET_FMT_plx " sh %d hash " TARGET_FMT_plx

                    " mask " TARGET_FMT_plx " " TARGET_FMT_lx "\n",

                    sdr, sdr_sh, hash, mask, page_mask);

            ctx->pg_addr[0] = get_pgaddr(sdr, sdr_sh, hash, mask);

            /* Secondary table address */

            hash = (~hash) & vsid_mask;

            LOG_MMU("sdr " TARGET_FMT_plx " sh %d hash " TARGET_FMT_plx

                    " mask " TARGET_FMT_plx "\n", sdr, sdr_sh, hash, mask);

            ctx->pg_addr[1] = get_pgaddr(sdr, sdr_sh, hash, mask);

#if defined(TARGET_PPC64)

            if (env->mmu_model & POWERPC_MMU_64) {

                /* Only 5 bits of the page index are used in the AVPN */

                if (target_page_bits > 23) {

                    ctx->ptem = (vsid << 12) |

                                ((pgidx << (target_page_bits - 16)) & 0xF80);

                } else {

                    ctx->ptem = (vsid << 12) | ((pgidx >> 4) & 0x0F80);

                }

            } else

#endif

            {

                ctx->ptem = (vsid << 7) | (pgidx >> 10);

            }

            /* Initialize real address with an invalid value */

            ctx->raddr = (target_phys_addr_t)-1ULL;

            if (unlikely(env->mmu_model == POWERPC_MMU_SOFT_6xx ||

                         env->mmu_model == POWERPC_MMU_SOFT_74xx)) {

                /* Software TLB search */

                ret = ppc6xx_tlb_check(env, ctx, eaddr, rw, type);

            } else {

                LOG_MMU("0 sdr1=" TARGET_FMT_plx " vsid=" TARGET_FMT_lx " "

                        "api=" TARGET_FMT_lx " hash=" TARGET_FMT_plx

                        " pg_addr=" TARGET_FMT_plx "\n",

                        sdr, vsid, pgidx, hash, ctx->pg_addr[0]);

                /* Primary table lookup */

                ret = find_pte(env, ctx, 0, rw, type, target_page_bits);

                if (ret < 0) {

                    /* Secondary table lookup */

                    if (eaddr != 0xEFFFFFFF)

                        LOG_MMU("1 sdr1=" TARGET_FMT_plx " vsid=" TARGET_FMT_lx " "

                                "api=" TARGET_FMT_lx " hash=" TARGET_FMT_plx

                                " pg_addr=" TARGET_FMT_plx "\n", sdr, vsid,

                                pgidx, hash, ctx->pg_addr[1]);

                    ret2 = find_pte(env, ctx, 1, rw, type,

                                    target_page_bits);

                    if (ret2 != -1)

                        ret = ret2;

                }

            }

#if defined (DUMP_PAGE_TABLES)

            if (qemu_log_enabled()) {

                target_phys_addr_t curaddr;

                uint32_t a0, a1, a2, a3;

                qemu_log("Page table: " TARGET_FMT_plx " len " TARGET_FMT_plx

                         "\n", sdr, mask + 0x80);

                for (curaddr = sdr; curaddr < (sdr + mask + 0x80);

                     curaddr += 16) {

                    a0 = ldl_phys(curaddr);

                    a1 = ldl_phys(curaddr + 4);

                    a2 = ldl_phys(curaddr + 8);

                    a3 = ldl_phys(curaddr + 12);

                    if (a0 != 0 || a1 != 0 || a2 != 0 || a3 != 0) {

                        qemu_log(TARGET_FMT_plx ": %08x %08x %08x %08x\n",

                                 curaddr, a0, a1, a2, a3);

                    }

                }

            }

#endif

        } else {

            LOG_MMU("No access allowed\n");

            ret = -3;

        }

    } else {

        LOG_MMU("direct store...\n");

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

            qemu_log("ERROR: instruction should not need "

                        "address translation\n");

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
