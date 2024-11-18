void cpu_ppc_reset (void *opaque)

{

    CPUPPCState *env;

    target_ulong msr;



    env = opaque;

    msr = (target_ulong)0;

#if defined(TARGET_PPC64)

    msr |= (target_ulong)0 << MSR_HV; /* Should be 1... */

#endif

    msr |= (target_ulong)0 << MSR_AP; /* TO BE CHECKED */

    msr |= (target_ulong)0 << MSR_SA; /* TO BE CHECKED */

    msr |= (target_ulong)1 << MSR_EP;

#if defined (DO_SINGLE_STEP) && 0

    /* Single step trace mode */

    msr |= (target_ulong)1 << MSR_SE;

    msr |= (target_ulong)1 << MSR_BE;

#endif

#if defined(CONFIG_USER_ONLY)

    msr |= (target_ulong)1 << MSR_FP; /* Allow floating point usage */

    msr |= (target_ulong)1 << MSR_PR;

#else

    env->nip = env->hreset_vector | env->excp_prefix;

    if (env->mmu_model != POWERPC_MMU_REAL_4xx)

        ppc_tlb_invalidate_all(env);

#endif

    env->msr = msr;

    hreg_compute_hflags(env);

    env->reserve = -1;

    /* Be sure no exception or interrupt is pending */

    env->pending_interrupts = 0;

    env->exception_index = POWERPC_EXCP_NONE;

    env->error_code = 0;

    /* Flush all TLBs */

    tlb_flush(env, 1);

}
