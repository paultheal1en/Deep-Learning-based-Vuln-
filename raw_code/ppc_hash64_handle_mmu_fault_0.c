int ppc_hash64_handle_mmu_fault(CPUPPCState *env, target_ulong address, int rw,

                                int mmu_idx)

{

    struct mmu_ctx_hash64 ctx;

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

    ret = ppc_hash64_get_physical_address(env, &ctx, address, rw, access_type);

    if (ret == 0) {

        tlb_set_page(env, address & TARGET_PAGE_MASK,

                     ctx.raddr & TARGET_PAGE_MASK, ctx.prot,

                     mmu_idx, TARGET_PAGE_SIZE);

        ret = 0;

    } else if (ret < 0) {

        LOG_MMU_STATE(env);

        if (access_type == ACCESS_CODE) {

            switch (ret) {

            case -1:

                env->exception_index = POWERPC_EXCP_ISI;

                env->error_code = 0x40000000;

                break;

            case -2:

                /* Access rights violation */

                env->exception_index = POWERPC_EXCP_ISI;

                env->error_code = 0x08000000;

                break;

            case -3:

                /* No execute protection violation */

                env->exception_index = POWERPC_EXCP_ISI;

                env->error_code = 0x10000000;

                break;

            case -5:

                /* No match in segment table */

                env->exception_index = POWERPC_EXCP_ISEG;

                env->error_code = 0;

                break;

            }

        } else {

            switch (ret) {

            case -1:

                /* No matches in page tables or TLB */

                env->exception_index = POWERPC_EXCP_DSI;

                env->error_code = 0;

                env->spr[SPR_DAR] = address;

                if (rw == 1) {

                    env->spr[SPR_DSISR] = 0x42000000;

                } else {

                    env->spr[SPR_DSISR] = 0x40000000;

                }

                break;

            case -2:

                /* Access rights violation */

                env->exception_index = POWERPC_EXCP_DSI;

                env->error_code = 0;

                env->spr[SPR_DAR] = address;

                if (rw == 1) {

                    env->spr[SPR_DSISR] = 0x0A000000;

                } else {

                    env->spr[SPR_DSISR] = 0x08000000;

                }

                break;

            case -5:

                /* No match in segment table */

                env->exception_index = POWERPC_EXCP_DSEG;

                env->error_code = 0;

                env->spr[SPR_DAR] = address;

                break;

            }

        }

#if 0

        printf("%s: set exception to %d %02x\n", __func__,

               env->exception, env->error_code);

#endif

        ret = 1;

    }



    return ret;

}
