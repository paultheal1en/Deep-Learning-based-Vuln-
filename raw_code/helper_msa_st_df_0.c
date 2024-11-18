void helper_msa_st_df(CPUMIPSState *env, uint32_t df, uint32_t wd, uint32_t rs,

                     int32_t s10)

{

    wr_t *pwd = &(env->active_fpu.fpr[wd].wr);

    target_ulong addr = env->active_tc.gpr[rs] + (s10 << df);

    int i;



    switch (df) {

    case DF_BYTE:

        for (i = 0; i < DF_ELEMENTS(DF_BYTE); i++) {

            do_sb(env, addr + (i << DF_BYTE), pwd->b[i],

                    env->hflags & MIPS_HFLAG_KSU);

        }

        break;

    case DF_HALF:

        for (i = 0; i < DF_ELEMENTS(DF_HALF); i++) {

            do_sh(env, addr + (i << DF_HALF), pwd->h[i],

                    env->hflags & MIPS_HFLAG_KSU);

        }

        break;

    case DF_WORD:

        for (i = 0; i < DF_ELEMENTS(DF_WORD); i++) {

            do_sw(env, addr + (i << DF_WORD), pwd->w[i],

                    env->hflags & MIPS_HFLAG_KSU);

        }

        break;

    case DF_DOUBLE:

        for (i = 0; i < DF_ELEMENTS(DF_DOUBLE); i++) {

            do_sd(env, addr + (i << DF_DOUBLE), pwd->d[i],

                    env->hflags & MIPS_HFLAG_KSU);

        }

        break;

    }

}
