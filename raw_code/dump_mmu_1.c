void dump_mmu(FILE *f, fprintf_function cpu_fprintf, CPUPPCState *env)

{

    switch (env->mmu_model) {

    case POWERPC_MMU_BOOKE:

        mmubooke_dump_mmu(f, cpu_fprintf, env);

        break;

    case POWERPC_MMU_BOOKE206:

        mmubooke206_dump_mmu(f, cpu_fprintf, env);

        break;

    case POWERPC_MMU_SOFT_6xx:

    case POWERPC_MMU_SOFT_74xx:

        mmu6xx_dump_mmu(f, cpu_fprintf, env);

        break;

#if defined(TARGET_PPC64)

    case POWERPC_MMU_64B:

    case POWERPC_MMU_2_03:

    case POWERPC_MMU_2_06:


    case POWERPC_MMU_2_07:


        dump_slb(f, cpu_fprintf, env);

        break;

#endif

    default:

        qemu_log_mask(LOG_UNIMP, "%s: unimplemented\n", __func__);

    }

}