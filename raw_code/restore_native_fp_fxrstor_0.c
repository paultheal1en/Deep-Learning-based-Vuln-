static void restore_native_fp_fxrstor(CPUState *env)

{

    struct fpxstate *fp = &fpx1;

    int i, j, fptag;



    fp->fpuc = env->fpuc;

    fp->fpus = (env->fpus & ~0x3800) | (env->fpstt & 0x7) << 11;

    fptag = 0;

    for(i = 0; i < 8; i++)

        fptag |= (env->fptags[i] << i);

    fp->fptag = fptag ^ 0xff;



    j = env->fpstt;

    for(i = 0;i < 8; i++) {

        memcpy(&fp->fpregs1[i * 16], &env->fpregs[j].d, 10);

        j = (j + 1) & 7;

    }

    if (env->cpuid_features & CPUID_SSE) {

        fp->mxcsr = env->mxcsr;

        /* XXX: check if DAZ is not available */

        fp->mxcsr_mask = 0xffff;

        memcpy(fp->xmm_regs, env->xmm_regs, CPU_NB_REGS * 16);

    }

    asm volatile ("fxrstor %0" : "=m" (*fp));

}
