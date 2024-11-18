void cpu_dump_state (CPUState *env, FILE *f, 

                     int (*cpu_fprintf)(FILE *f, const char *fmt, ...),

                     int flags)

{

    uint32_t c0_status;

    int i;

    

    cpu_fprintf(f, "pc=0x" TARGET_FMT_lx " HI=0x" TARGET_FMT_lx " LO=0x" TARGET_FMT_lx " ds %04x " TARGET_FMT_lx " %d\n",

                env->PC, env->HI, env->LO, env->hflags, env->btarget, env->bcond);

    for (i = 0; i < 32; i++) {

        if ((i & 3) == 0)

            cpu_fprintf(f, "GPR%02d:", i);

        cpu_fprintf(f, " %s " TARGET_FMT_lx, regnames[i], env->gpr[i]);

        if ((i & 3) == 3)

            cpu_fprintf(f, "\n");

    }



    c0_status = env->CP0_Status;



    cpu_fprintf(f, "CP0 Status  0x%08x Cause   0x%08x EPC    0x" TARGET_FMT_lx "\n",

                c0_status, env->CP0_Cause, env->CP0_EPC);

    cpu_fprintf(f, "    Config0 0x%08x Config1 0x%08x LLAddr 0x" TARGET_FMT_lx "\n",

                env->CP0_Config0, env->CP0_Config1, env->CP0_LLAddr);

    if (c0_status & (1 << CP0St_CU1))

        fpu_dump_state(env, f, cpu_fprintf, flags);

#if defined(TARGET_MIPS64) && defined(MIPS_DEBUG_SIGN_EXTENSIONS)

    cpu_mips_check_sign_extensions(env, f, cpu_fprintf, flags);

#endif

}
