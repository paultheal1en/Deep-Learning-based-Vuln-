static inline TranslationBlock *tb_find_fast(void)

{

    TranslationBlock *tb;

    target_ulong cs_base, pc;

    uint64_t flags;



    /* we record a subset of the CPU state. It will

       always be the same before a given translated block

       is executed. */

#if defined(TARGET_I386)

    flags = env->hflags;

    flags |= (env->eflags & (IOPL_MASK | TF_MASK | VM_MASK));

    flags |= env->intercept;

    cs_base = env->segs[R_CS].base;

    pc = cs_base + env->eip;

#elif defined(TARGET_ARM)

    flags = env->thumb | (env->vfp.vec_len << 1)

            | (env->vfp.vec_stride << 4);

    if ((env->uncached_cpsr & CPSR_M) != ARM_CPU_MODE_USR)

        flags |= (1 << 6);

    if (env->vfp.xregs[ARM_VFP_FPEXC] & (1 << 30))

        flags |= (1 << 7);

    flags |= (env->condexec_bits << 8);

    cs_base = 0;

    pc = env->regs[15];

#elif defined(TARGET_SPARC)

#ifdef TARGET_SPARC64

    // Combined FPU enable bits . PRIV . DMMU enabled . IMMU enabled

    flags = (((env->pstate & PS_PEF) >> 1) | ((env->fprs & FPRS_FEF) << 2))

        | (env->pstate & PS_PRIV) | ((env->lsu & (DMMU_E | IMMU_E)) >> 2);

#else

    // FPU enable . Supervisor

    flags = (env->psref << 4) | env->psrs;

#endif

    cs_base = env->npc;

    pc = env->pc;

#elif defined(TARGET_PPC)

    flags = env->hflags;

    cs_base = 0;

    pc = env->nip;

#elif defined(TARGET_MIPS)

    flags = env->hflags & (MIPS_HFLAG_TMASK | MIPS_HFLAG_BMASK);

    cs_base = 0;

    pc = env->PC[env->current_tc];

#elif defined(TARGET_M68K)

    flags = (env->fpcr & M68K_FPCR_PREC)  /* Bit  6 */

            | (env->sr & SR_S)            /* Bit  13 */

            | ((env->macsr >> 4) & 0xf);  /* Bits 0-3 */

    cs_base = 0;

    pc = env->pc;

#elif defined(TARGET_SH4)

    flags = env->flags;

    cs_base = 0;

    pc = env->pc;

#elif defined(TARGET_ALPHA)

    flags = env->ps;

    cs_base = 0;

    pc = env->pc;

#elif defined(TARGET_CRIS)

    flags = env->pregs[PR_CCS] & (U_FLAG | X_FLAG);

    cs_base = 0;

    pc = env->pc;

#else

#error unsupported CPU

#endif

    tb = env->tb_jmp_cache[tb_jmp_cache_hash_func(pc)];

    if (__builtin_expect(!tb || tb->pc != pc || tb->cs_base != cs_base ||

                         tb->flags != flags, 0)) {

        tb = tb_find_slow(pc, cs_base, flags);

        /* Note: we do it here to avoid a gcc bug on Mac OS X when

           doing it in tb_find_slow */

        if (tb_invalidated_flag) {

            /* as some TB could have been invalidated because

               of memory exceptions while generating the code, we

               must recompute the hash index here */

            T0 = 0;

        }

    }

    return tb;

}
