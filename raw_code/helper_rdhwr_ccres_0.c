target_ulong helper_rdhwr_ccres(CPUMIPSState *env)

{

    if ((env->hflags & MIPS_HFLAG_CP0) ||

        (env->CP0_HWREna & (1 << 3)))

        return env->CCRes;

    else

        do_raise_exception(env, EXCP_RI, GETPC());



    return 0;

}
