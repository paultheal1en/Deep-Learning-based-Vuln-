static void save_native_fp_fxsave(CPUState *env)

{

    struct fpxstate *fp = &fpx1;

    int fptag, i, j;

    uint16_t fpuc;



    asm volatile ("fxsave %0" : : "m" (*fp));

    env->fpuc = fp->fpuc;

    env->fpstt = (fp->fpus >> 11) & 7;

    env->fpus = fp->fpus & ~0x3800;

    fptag = fp->fptag ^ 0xff;

    for(i = 0;i < 8; i++) {

        env->fptags[i] = (fptag >> i) & 1;

    }

    j = env->fpstt;

    for(i = 0;i < 8; i++) {

        memcpy(&env->fpregs[j].d, &fp->fpregs1[i * 16], 10);

        j = (j + 1) & 7;

    }

    if (env->cpuid_features & CPUID_SSE) {

        env->mxcsr = fp->mxcsr;

        memcpy(env->xmm_regs, fp->xmm_regs, CPU_NB_REGS * 16);

    }



    /* we must restore the default rounding state */

    asm volatile ("fninit");

    fpuc = 0x037f | (env->fpuc & (3 << 10));

    asm volatile("fldcw %0" : : "m" (fpuc));

}
