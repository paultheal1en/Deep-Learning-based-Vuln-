int get_segment64(CPUPPCState *env, mmu_ctx_t *ctx,

                  target_ulong eaddr, int rw, int type)

{

    hwaddr hash;

    target_ulong vsid;

    int pr, target_page_bits;

    int ret, ret2;



    pr = msr_pr;

    ctx->eaddr = eaddr;

    ppc_slb_t *slb;

    target_ulong pageaddr;

    int segment_bits;



    LOG_MMU("Check SLBs\n");

    slb = slb_lookup(env, eaddr);

    if (!slb) {

        return -5;

    }



    if (slb->vsid & SLB_VSID_B) {

        vsid = (slb->vsid & SLB_VSID_VSID) >> SLB_VSID_SHIFT_1T;

        segment_bits = 40;

    } else {

        vsid = (slb->vsid & SLB_VSID_VSID) >> SLB_VSID_SHIFT;

        segment_bits = 28;

    }



    target_page_bits = (slb->vsid & SLB_VSID_L)

        ? TARGET_PAGE_BITS_16M : TARGET_PAGE_BITS;

    ctx->key = !!(pr ? (slb->vsid & SLB_VSID_KP)

                  : (slb->vsid & SLB_VSID_KS));

    ctx->nx = !!(slb->vsid & SLB_VSID_N);



    pageaddr = eaddr & ((1ULL << segment_bits)

                            - (1ULL << target_page_bits));

    if (slb->vsid & SLB_VSID_B) {

        hash = vsid ^ (vsid << 25) ^ (pageaddr >> target_page_bits);

    } else {

        hash = vsid ^ (pageaddr >> target_page_bits);

    }

    /* Only 5 bits of the page index are used in the AVPN */

    ctx->ptem = (slb->vsid & SLB_VSID_PTEM) |

        ((pageaddr >> 16) & ((1ULL << segment_bits) - 0x80));



    LOG_MMU("pte segment: key=%d nx %d vsid " TARGET_FMT_lx "\n",

            ctx->key, ctx->nx, vsid);

    ret = -1;



    /* Check if instruction fetch is allowed, if needed */

    if (type != ACCESS_CODE || ctx->nx == 0) {

        /* Page address translation */

        LOG_MMU("htab_base " TARGET_FMT_plx " htab_mask " TARGET_FMT_plx

                " hash " TARGET_FMT_plx "\n",

                env->htab_base, env->htab_mask, hash);

        ctx->hash[0] = hash;

        ctx->hash[1] = ~hash;



        /* Initialize real address with an invalid value */

        ctx->raddr = (hwaddr)-1ULL;

        LOG_MMU("0 htab=" TARGET_FMT_plx "/" TARGET_FMT_plx

                " vsid=" TARGET_FMT_lx " ptem=" TARGET_FMT_lx

                " hash=" TARGET_FMT_plx "\n",

                env->htab_base, env->htab_mask, vsid, ctx->ptem,

                ctx->hash[0]);

        /* Primary table lookup */

        ret = find_pte64(env, ctx, 0, rw, type, target_page_bits);

        if (ret < 0) {

            /* Secondary table lookup */

            LOG_MMU("1 htab=" TARGET_FMT_plx "/" TARGET_FMT_plx

                    " vsid=" TARGET_FMT_lx " api=" TARGET_FMT_lx

                    " hash=" TARGET_FMT_plx "\n", env->htab_base,

                    env->htab_mask, vsid, ctx->ptem, ctx->hash[1]);

            ret2 = find_pte64(env, ctx, 1, rw, type, target_page_bits);

            if (ret2 != -1) {

                ret = ret2;

            }

        }

    } else {

        LOG_MMU("No access allowed\n");

        ret = -3;

    }



    return ret;

}
