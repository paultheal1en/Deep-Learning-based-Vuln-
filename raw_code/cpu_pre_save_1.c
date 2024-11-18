static void cpu_pre_save(void *opaque)

{

    CPUState *env = opaque;

    int i;



    cpu_synchronize_state(env);



    /* FPU */

    env->fpus_vmstate = (env->fpus & ~0x3800) | (env->fpstt & 0x7) << 11;

    env->fptag_vmstate = 0;

    for(i = 0; i < 8; i++) {

        env->fptag_vmstate |= ((!env->fptags[i]) << i);

    }



#ifdef USE_X86LDOUBLE

    env->fpregs_format_vmstate = 0;

#else

    env->fpregs_format_vmstate = 1;

#endif

}
