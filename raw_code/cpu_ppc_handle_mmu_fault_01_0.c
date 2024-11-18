static int cpu_ppc_handle_mmu_fault(CPUPPCState *env, target_ulong address,

                                    int rw, int mmu_idx)

{

    CPUState *cs = CPU(ppc_env_get_cpu(env));

    PowerPCCPU *cpu = POWERPC_CPU(cs);

    mmu_ctx_t ctx;

    int access_type;

    int ret = 0;



    if (rw == 2) {

        /* code access */

        rw = 0;

        access_type = ACCESS_CODE;

    } else {

        /* data access */

        access_type = env->access_type;

    }

    ret = get_physical_address(env, &ctx, address, rw, access_type);

    if (ret == 0) {

        tlb_set_page(cs, address & TARGET_PAGE_MASK,

                     ctx.raddr & TARGET_PAGE_MASK, ctx.prot,

                     mmu_idx, TARGET_PAGE_SIZE);

        ret = 0;

    } else if (ret < 0) {

        LOG_MMU_STATE(cs);

        if (access_type == ACCESS_CODE) {

            switch (ret) {

            case -1:

                /* No matches in page tables or TLB */

                switch (env->mmu_model) {

                case POWERPC_MMU_SOFT_6xx:

                    cs->exception_index = POWERPC_EXCP_IFTLB;

                    env->error_code = 1 << 18;

                    env->spr[SPR_IMISS] = address;

                    env->spr[SPR_ICMP] = 0x80000000 | ctx.ptem;

                    goto tlb_miss;

                case POWERPC_MMU_SOFT_74xx:

                    cs->exception_index = POWERPC_EXCP_IFTLB;

                    goto tlb_miss_74xx;

                case POWERPC_MMU_SOFT_4xx:

                case POWERPC_MMU_SOFT_4xx_Z:

                    cs->exception_index = POWERPC_EXCP_ITLB;

                    env->error_code = 0;

                    env->spr[SPR_40x_DEAR] = address;

                    env->spr[SPR_40x_ESR] = 0x00000000;

                    break;

                case POWERPC_MMU_BOOKE206:

                    booke206_update_mas_tlb_miss(env, address, rw);

                    /* fall through */

                case POWERPC_MMU_BOOKE:

                    cs->exception_index = POWERPC_EXCP_ITLB;

                    env->error_code = 0;

                    env->spr[SPR_BOOKE_DEAR] = address;

                    return -1;

                case POWERPC_MMU_MPC8xx:

                    /* XXX: TODO */

                    cpu_abort(cs, "MPC8xx MMU model is not implemented\n");

                    break;

                case POWERPC_MMU_REAL:

                    cpu_abort(cs, "PowerPC in real mode should never raise "

                              "any MMU exceptions\n");

                    return -1;

                default:

                    cpu_abort(cs, "Unknown or invalid MMU model\n");

                    return -1;

                }

                break;

            case -2:

                /* Access rights violation */

                cs->exception_index = POWERPC_EXCP_ISI;

                env->error_code = 0x08000000;

                break;

            case -3:

                /* No execute protection violation */

                if ((env->mmu_model == POWERPC_MMU_BOOKE) ||

                    (env->mmu_model == POWERPC_MMU_BOOKE206)) {

                    env->spr[SPR_BOOKE_ESR] = 0x00000000;

                }

                cs->exception_index = POWERPC_EXCP_ISI;

                env->error_code = 0x10000000;

                break;

            case -4:

                /* Direct store exception */

                /* No code fetch is allowed in direct-store areas */

                cs->exception_index = POWERPC_EXCP_ISI;

                env->error_code = 0x10000000;

                break;

            }

        } else {

            switch (ret) {

            case -1:

                /* No matches in page tables or TLB */

                switch (env->mmu_model) {

                case POWERPC_MMU_SOFT_6xx:

                    if (rw == 1) {

                        cs->exception_index = POWERPC_EXCP_DSTLB;

                        env->error_code = 1 << 16;

                    } else {

                        cs->exception_index = POWERPC_EXCP_DLTLB;

                        env->error_code = 0;

                    }

                    env->spr[SPR_DMISS] = address;

                    env->spr[SPR_DCMP] = 0x80000000 | ctx.ptem;

                tlb_miss:

                    env->error_code |= ctx.key << 19;

                    env->spr[SPR_HASH1] = env->htab_base +

                        get_pteg_offset32(cpu, ctx.hash[0]);

                    env->spr[SPR_HASH2] = env->htab_base +

                        get_pteg_offset32(cpu, ctx.hash[1]);

                    break;

                case POWERPC_MMU_SOFT_74xx:

                    if (rw == 1) {

                        cs->exception_index = POWERPC_EXCP_DSTLB;

                    } else {

                        cs->exception_index = POWERPC_EXCP_DLTLB;

                    }

                tlb_miss_74xx:

                    /* Implement LRU algorithm */

                    env->error_code = ctx.key << 19;

                    env->spr[SPR_TLBMISS] = (address & ~((target_ulong)0x3)) |

                        ((env->last_way + 1) & (env->nb_ways - 1));

                    env->spr[SPR_PTEHI] = 0x80000000 | ctx.ptem;

                    break;

                case POWERPC_MMU_SOFT_4xx:

                case POWERPC_MMU_SOFT_4xx_Z:

                    cs->exception_index = POWERPC_EXCP_DTLB;

                    env->error_code = 0;

                    env->spr[SPR_40x_DEAR] = address;

                    if (rw) {

                        env->spr[SPR_40x_ESR] = 0x00800000;

                    } else {

                        env->spr[SPR_40x_ESR] = 0x00000000;

                    }

                    break;

                case POWERPC_MMU_MPC8xx:

                    /* XXX: TODO */

                    cpu_abort(cs, "MPC8xx MMU model is not implemented\n");

                    break;

                case POWERPC_MMU_BOOKE206:

                    booke206_update_mas_tlb_miss(env, address, rw);

                    /* fall through */

                case POWERPC_MMU_BOOKE:

                    cs->exception_index = POWERPC_EXCP_DTLB;

                    env->error_code = 0;

                    env->spr[SPR_BOOKE_DEAR] = address;

                    env->spr[SPR_BOOKE_ESR] = rw ? ESR_ST : 0;

                    return -1;

                case POWERPC_MMU_REAL:

                    cpu_abort(cs, "PowerPC in real mode should never raise "

                              "any MMU exceptions\n");

                    return -1;

                default:

                    cpu_abort(cs, "Unknown or invalid MMU model\n");

                    return -1;

                }

                break;

            case -2:

                /* Access rights violation */

                cs->exception_index = POWERPC_EXCP_DSI;

                env->error_code = 0;

                if (env->mmu_model == POWERPC_MMU_SOFT_4xx

                    || env->mmu_model == POWERPC_MMU_SOFT_4xx_Z) {

                    env->spr[SPR_40x_DEAR] = address;

                    if (rw) {

                        env->spr[SPR_40x_ESR] |= 0x00800000;

                    }

                } else if ((env->mmu_model == POWERPC_MMU_BOOKE) ||

                           (env->mmu_model == POWERPC_MMU_BOOKE206)) {

                    env->spr[SPR_BOOKE_DEAR] = address;

                    env->spr[SPR_BOOKE_ESR] = rw ? ESR_ST : 0;

                } else {

                    env->spr[SPR_DAR] = address;

                    if (rw == 1) {

                        env->spr[SPR_DSISR] = 0x0A000000;

                    } else {

                        env->spr[SPR_DSISR] = 0x08000000;

                    }

                }

                break;

            case -4:

                /* Direct store exception */

                switch (access_type) {

                case ACCESS_FLOAT:

                    /* Floating point load/store */

                    cs->exception_index = POWERPC_EXCP_ALIGN;

                    env->error_code = POWERPC_EXCP_ALIGN_FP;

                    env->spr[SPR_DAR] = address;

                    break;

                case ACCESS_RES:

                    /* lwarx, ldarx or stwcx. */

                    cs->exception_index = POWERPC_EXCP_DSI;

                    env->error_code = 0;

                    env->spr[SPR_DAR] = address;

                    if (rw == 1) {

                        env->spr[SPR_DSISR] = 0x06000000;

                    } else {

                        env->spr[SPR_DSISR] = 0x04000000;

                    }

                    break;

                case ACCESS_EXT:

                    /* eciwx or ecowx */

                    cs->exception_index = POWERPC_EXCP_DSI;

                    env->error_code = 0;

                    env->spr[SPR_DAR] = address;

                    if (rw == 1) {

                        env->spr[SPR_DSISR] = 0x06100000;

                    } else {

                        env->spr[SPR_DSISR] = 0x04100000;

                    }

                    break;

                default:

                    printf("DSI: invalid exception (%d)\n", ret);

                    cs->exception_index = POWERPC_EXCP_PROGRAM;

                    env->error_code =

                        POWERPC_EXCP_INVAL | POWERPC_EXCP_INVAL_INVAL;

                    env->spr[SPR_DAR] = address;

                    break;

                }

                break;

            }

        }

#if 0

        printf("%s: set exception to %d %02x\n", __func__,

               cs->exception, env->error_code);

#endif

        ret = 1;

    }



    return ret;

}
