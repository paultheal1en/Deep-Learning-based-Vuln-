void ppc_tlb_invalidate_all(CPUPPCState *env)

{

    switch (env->mmu_model) {

    case POWERPC_MMU_SOFT_6xx:

    case POWERPC_MMU_SOFT_74xx:

        ppc6xx_tlb_invalidate_all(env);

        break;

    case POWERPC_MMU_SOFT_4xx:

    case POWERPC_MMU_SOFT_4xx_Z:

        ppc4xx_tlb_invalidate_all(env);

        break;

    case POWERPC_MMU_REAL:

        cpu_abort(env, "No TLB for PowerPC 4xx in real mode\n");

        break;

    case POWERPC_MMU_MPC8xx:

        /* XXX: TODO */

        cpu_abort(env, "MPC8xx MMU model is not implemented\n");

        break;

    case POWERPC_MMU_BOOKE:

        tlb_flush(env, 1);

        break;

    case POWERPC_MMU_BOOKE206:

        booke206_flush_tlb(env, -1, 0);

        break;

    case POWERPC_MMU_32B:

    case POWERPC_MMU_601:

#if defined(TARGET_PPC64)

    case POWERPC_MMU_620:

    case POWERPC_MMU_64B:

    case POWERPC_MMU_2_06:


#endif /* defined(TARGET_PPC64) */

        tlb_flush(env, 1);

        break;

    default:

        /* XXX: TODO */

        cpu_abort(env, "Unknown MMU model\n");

        break;

    }

}