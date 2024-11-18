static int restore_user_regs(CPUPPCState *env,

                             struct target_mcontext *frame, int sig)

{

    target_ulong save_r2 = 0;

    target_ulong msr;

    target_ulong ccr;



    int i;



    if (!sig) {

        save_r2 = env->gpr[2];

    }



    /* Restore general registers.  */

    for (i = 0; i < ARRAY_SIZE(env->gpr); i++) {

        if (__get_user(env->gpr[i], &frame->mc_gregs[i])) {

            return 1;

        }

    }

    if (__get_user(env->nip, &frame->mc_gregs[TARGET_PT_NIP])

        || __get_user(env->ctr, &frame->mc_gregs[TARGET_PT_CTR])

        || __get_user(env->lr, &frame->mc_gregs[TARGET_PT_LNK])

        || __get_user(env->xer, &frame->mc_gregs[TARGET_PT_XER]))

        return 1;

    if (__get_user(ccr, &frame->mc_gregs[TARGET_PT_CCR]))

        return 1;



    for (i = 0; i < ARRAY_SIZE(env->crf); i++) {

        env->crf[i] = (ccr >> (32 - ((i + 1) * 4))) & 0xf;

    }



    if (!sig) {

        env->gpr[2] = save_r2;

    }

    /* Restore MSR.  */

    if (__get_user(msr, &frame->mc_gregs[TARGET_PT_MSR]))

        return 1;



    /* If doing signal return, restore the previous little-endian mode.  */

    if (sig)

        env->msr = (env->msr & ~MSR_LE) | (msr & MSR_LE);



    /* Restore Altivec registers if necessary.  */

    if (env->insns_flags & PPC_ALTIVEC) {

        for (i = 0; i < ARRAY_SIZE(env->avr); i++) {

            ppc_avr_t *avr = &env->avr[i];

            ppc_avr_t *vreg = &frame->mc_vregs.altivec[i];



            if (__get_user(avr->u64[0], &vreg->u64[0]) ||

                __get_user(avr->u64[1], &vreg->u64[1])) {

                return 1;

            }

        }

        /* Set MSR_VEC in the saved MSR value to indicate that

           frame->mc_vregs contains valid data.  */

        if (__get_user(env->spr[SPR_VRSAVE],

                       (target_ulong *)(&frame->mc_vregs.altivec[32].u32[3])))

            return 1;

    }



    /* Restore floating point registers.  */

    if (env->insns_flags & PPC_FLOAT) {

        uint64_t fpscr;

        for (i = 0; i < ARRAY_SIZE(env->fpr); i++) {

            if (__get_user(env->fpr[i], &frame->mc_fregs[i])) {

                return 1;

            }

        }

        if (__get_user(fpscr, &frame->mc_fregs[32]))

            return 1;

        env->fpscr = (uint32_t) fpscr;

    }



    /* Save SPE registers.  The kernel only saves the high half.  */

    if (env->insns_flags & PPC_SPE) {

#if defined(TARGET_PPC64)

        for (i = 0; i < ARRAY_SIZE(env->gpr); i++) {

            uint32_t hi;



            if (__get_user(hi, &frame->mc_vregs.spe[i])) {

                return 1;

            }

            env->gpr[i] = ((uint64_t)hi << 32) | ((uint32_t) env->gpr[i]);

        }

#else

        for (i = 0; i < ARRAY_SIZE(env->gprh); i++) {

            if (__get_user(env->gprh[i], &frame->mc_vregs.spe[i])) {

                return 1;

            }

        }

#endif

        if (__get_user(env->spe_fscr, &frame->mc_vregs.spe[32]))

            return 1;

    }



    return 0;

}
