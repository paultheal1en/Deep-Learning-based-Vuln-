void helper_fxrstor(CPUX86State *env, target_ulong ptr, int data64)

{

    int i, fpus, fptag, nb_xmm_regs;

    floatx80 tmp;

    target_ulong addr;



    /* The operand must be 16 byte aligned */

    if (ptr & 0xf) {

        raise_exception(env, EXCP0D_GPF);

    }



    env->fpuc = cpu_lduw_data(env, ptr);

    fpus = cpu_lduw_data(env, ptr + 2);

    fptag = cpu_lduw_data(env, ptr + 4);

    env->fpstt = (fpus >> 11) & 7;

    env->fpus = fpus & ~0x3800;

    fptag ^= 0xff;

    for (i = 0; i < 8; i++) {

        env->fptags[i] = ((fptag >> i) & 1);

    }



    addr = ptr + 0x20;

    for (i = 0; i < 8; i++) {

        tmp = helper_fldt(env, addr);

        ST(i) = tmp;

        addr += 16;

    }



    if (env->cr[4] & CR4_OSFXSR_MASK) {

        /* XXX: finish it */

        env->mxcsr = cpu_ldl_data(env, ptr + 0x18);

        /* cpu_ldl_data(env, ptr + 0x1c); */

        if (env->hflags & HF_CS64_MASK) {

            nb_xmm_regs = 16;

        } else {

            nb_xmm_regs = 8;

        }

        addr = ptr + 0xa0;

        /* Fast FXRESTORE leaves out the XMM registers */

        if (!(env->efer & MSR_EFER_FFXSR)

            || (env->hflags & HF_CPL_MASK)

            || !(env->hflags & HF_LMA_MASK)) {

            for (i = 0; i < nb_xmm_regs; i++) {

                env->xmm_regs[i].XMM_Q(0) = cpu_ldq_data(env, addr);

                env->xmm_regs[i].XMM_Q(1) = cpu_ldq_data(env, addr + 8);

                addr += 16;

            }

        }

    }

}
