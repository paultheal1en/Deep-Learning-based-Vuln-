void mips_cpu_dump_state(CPUState *cs, FILE *f, fprintf_function cpu_fprintf,

                         int flags)

{

    MIPSCPU *cpu = MIPS_CPU(cs);

    CPUMIPSState *env = &cpu->env;

    int i;



    cpu_fprintf(f, "pc=0x" TARGET_FMT_lx " HI=0x" TARGET_FMT_lx

                " LO=0x" TARGET_FMT_lx " ds %04x "

                TARGET_FMT_lx " " TARGET_FMT_ld "\n",

                env->active_tc.PC, env->active_tc.HI[0], env->active_tc.LO[0],

                env->hflags, env->btarget, env->bcond);

    for (i = 0; i < 32; i++) {

        if ((i & 3) == 0)

            cpu_fprintf(f, "GPR%02d:", i);

        cpu_fprintf(f, " %s " TARGET_FMT_lx, regnames[i], env->active_tc.gpr[i]);

        if ((i & 3) == 3)

            cpu_fprintf(f, "\n");

    }



    cpu_fprintf(f, "CP0 Status  0x%08x Cause   0x%08x EPC    0x" TARGET_FMT_lx "\n",

                env->CP0_Status, env->CP0_Cause, env->CP0_EPC);

    cpu_fprintf(f, "    Config0 0x%08x Config1 0x%08x LLAddr 0x%016"

                PRIx64 "\n",

                env->CP0_Config0, env->CP0_Config1, env->lladdr);

    cpu_fprintf(f, "    Config2 0x%08x Config3 0x%08x\n",

                env->CP0_Config2, env->CP0_Config3);

    cpu_fprintf(f, "    Config4 0x%08x Config5 0x%08x\n",

                env->CP0_Config4, env->CP0_Config5);

    if (env->hflags & MIPS_HFLAG_FPU)

        fpu_dump_state(env, f, cpu_fprintf, flags);

#if defined(TARGET_MIPS64) && defined(MIPS_DEBUG_SIGN_EXTENSIONS)

    cpu_mips_check_sign_extensions(env, f, cpu_fprintf, flags);

#endif

}
