static void save_native_fp_fsave(CPUState *env)

{

    int fptag, i, j;

    uint16_t fpuc;

    struct fpstate fp1, *fp = &fp1;



    asm volatile ("fsave %0" : : "m" (*fp));

    env->fpuc = fp->fpuc;

    env->fpstt = (fp->fpus >> 11) & 7;

    env->fpus = fp->fpus & ~0x3800;

    fptag = fp->fptag;

    for(i = 0;i < 8; i++) {

        env->fptags[i] = ((fptag & 3) == 3);

        fptag >>= 2;

    }

    j = env->fpstt;

    for(i = 0;i < 8; i++) {

        memcpy(&env->fpregs[j].d, &fp->fpregs1[i * 10], 10);

        j = (j + 1) & 7;

    }

    /* we must restore the default rounding state */

    fpuc = 0x037f | (env->fpuc & (3 << 10));

    asm volatile("fldcw %0" : : "m" (fpuc));

}
