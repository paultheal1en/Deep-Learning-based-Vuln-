static int save_user_regs(CPUPPCState *env, struct target_mcontext *frame,

                          int sigret)

{

    target_ulong msr = env->msr;

    int i;

    target_ulong ccr = 0;



    /* In general, the kernel attempts to be intelligent about what it

       needs to save for Altivec/FP/SPE registers.  We don't care that

       much, so we just go ahead and save everything.  */



    /* Save general registers.  */

    for (i = 0; i < ARRAY_SIZE(env->gpr); i++) {

        if (__put_user(env->gpr[i], &frame->mc_gregs[i])) {

            return 1;

        }

    }

    if (__put_user(env->nip, &frame->mc_gregs[TARGET_PT_NIP])

        || __put_user(env->ctr, &frame->mc_gregs[TARGET_PT_CTR])

        || __put_user(env->lr, &frame->mc_gregs[TARGET_PT_LNK])

        || __put_user(env->xer, &frame->mc_gregs[TARGET_PT_XER]))

        return 1;



    for (i = 0; i < ARRAY_SIZE(env->crf); i++) {

        ccr |= env->crf[i] << (32 - ((i + 1) * 4));

    }

    if (__put_user(ccr, &frame->mc_gregs[TARGET_PT_CCR]))

        return 1;



    /* Save Altivec registers if necessary.  */

    if (env->insns_flags & PPC_ALTIVEC) {

        for (i = 0; i < ARRAY_SIZE(env->avr); i++) {

            ppc_avr_t *avr = &env->avr[i];

            ppc_avr_t *vreg = &frame->mc_vregs.altivec[i];



            if (__put_user(avr->u64[0], &vreg->u64[0]) ||

                __put_user(avr->u64[1], &vreg->u64[1])) {

                return 1;

            }

        }

        /* Set MSR_VR in the saved MSR value to indicate that

           frame->mc_vregs contains valid data.  */

        msr |= MSR_VR;

        if (__put_user((uint32_t)env->spr[SPR_VRSAVE],

                       &frame->mc_vregs.altivec[32].u32[3]))

            return 1;

    }



    /* Save floating point registers.  */

    if (env->insns_flags & PPC_FLOAT) {

        for (i = 0; i < ARRAY_SIZE(env->fpr); i++) {

            if (__put_user(env->fpr[i], &frame->mc_fregs[i])) {

                return 1;

            }

        }

        if (__put_user((uint64_t) env->fpscr, &frame->mc_fregs[32]))

            return 1;

    }



    /* Save SPE registers.  The kernel only saves the high half.  */

    if (env->insns_flags & PPC_SPE) {

#if defined(TARGET_PPC64)

        for (i = 0; i < ARRAY_SIZE(env->gpr); i++) {

            if (__put_user(env->gpr[i] >> 32, &frame->mc_vregs.spe[i])) {

                return 1;

            }

        }

#else

        for (i = 0; i < ARRAY_SIZE(env->gprh); i++) {

            if (__put_user(env->gprh[i], &frame->mc_vregs.spe[i])) {

                return 1;

            }

        }

#endif

        /* Set MSR_SPE in the saved MSR value to indicate that

           frame->mc_vregs contains valid data.  */

        msr |= MSR_SPE;

        if (__put_user(env->spe_fscr, &frame->mc_vregs.spe[32]))

            return 1;

    }



    /* Store MSR.  */

    if (__put_user(msr, &frame->mc_gregs[TARGET_PT_MSR]))

        return 1;



    /* Set up the sigreturn trampoline: li r0,sigret; sc.  */

    if (sigret) {

        if (__put_user(0x38000000UL | sigret, &frame->tramp[0]) ||

            __put_user(0x44000002UL, &frame->tramp[1])) {

            return 1;

        }

    }



    return 0;

}
