void cpu_dump_state (CPUState *env, FILE *f, fprintf_function cpu_fprintf,

                     int flags)

{

#define RGPL  4

#define RFPL  4



    int i;



    cpu_fprintf(f, "NIP " TARGET_FMT_lx "   LR " TARGET_FMT_lx " CTR "

                TARGET_FMT_lx " XER " TARGET_FMT_lx "\n",

                env->nip, env->lr, env->ctr, env->xer);

    cpu_fprintf(f, "MSR " TARGET_FMT_lx " HID0 " TARGET_FMT_lx "  HF "

                TARGET_FMT_lx " idx %d\n", env->msr, env->spr[SPR_HID0],

                env->hflags, env->mmu_idx);

#if !defined(NO_TIMER_DUMP)

    cpu_fprintf(f, "TB %08" PRIu32 " %08" PRIu64

#if !defined(CONFIG_USER_ONLY)

                " DECR %08" PRIu32

#endif

                "\n",

                cpu_ppc_load_tbu(env), cpu_ppc_load_tbl(env)

#if !defined(CONFIG_USER_ONLY)

                , cpu_ppc_load_decr(env)

#endif

                );

#endif

    for (i = 0; i < 32; i++) {

        if ((i & (RGPL - 1)) == 0)

            cpu_fprintf(f, "GPR%02d", i);

        cpu_fprintf(f, " %016" PRIx64, ppc_dump_gpr(env, i));

        if ((i & (RGPL - 1)) == (RGPL - 1))

            cpu_fprintf(f, "\n");

    }

    cpu_fprintf(f, "CR ");

    for (i = 0; i < 8; i++)

        cpu_fprintf(f, "%01x", env->crf[i]);

    cpu_fprintf(f, "  [");

    for (i = 0; i < 8; i++) {

        char a = '-';

        if (env->crf[i] & 0x08)

            a = 'L';

        else if (env->crf[i] & 0x04)

            a = 'G';

        else if (env->crf[i] & 0x02)

            a = 'E';

        cpu_fprintf(f, " %c%c", a, env->crf[i] & 0x01 ? 'O' : ' ');

    }

    cpu_fprintf(f, " ]             RES " TARGET_FMT_lx "\n",

                env->reserve_addr);

    for (i = 0; i < 32; i++) {

        if ((i & (RFPL - 1)) == 0)

            cpu_fprintf(f, "FPR%02d", i);

        cpu_fprintf(f, " %016" PRIx64, *((uint64_t *)&env->fpr[i]));

        if ((i & (RFPL - 1)) == (RFPL - 1))

            cpu_fprintf(f, "\n");

    }

    cpu_fprintf(f, "FPSCR %08x\n", env->fpscr);

#if !defined(CONFIG_USER_ONLY)

    cpu_fprintf(f, "SRR0 " TARGET_FMT_lx " SRR1 " TARGET_FMT_lx " SDR1 "

                TARGET_FMT_lx "\n", env->spr[SPR_SRR0], env->spr[SPR_SRR1],

                env->sdr1);

#endif



#undef RGPL

#undef RFPL

}
