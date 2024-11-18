static void unicore_ii_cpu_initfn(Object *obj)

{

    UniCore32CPU *cpu = UNICORE32_CPU(obj);

    CPUUniCore32State *env = &cpu->env;



    env->cp0.c0_cpuid = 0x40010863;



    set_feature(env, UC32_HWCAP_CMOV);

    set_feature(env, UC32_HWCAP_UCF64);

    env->ucf64.xregs[UC32_UCF64_FPSCR] = 0;

    env->cp0.c0_cachetype = 0x1dd20d2;

    env->cp0.c1_sys = 0x00090078;

}
