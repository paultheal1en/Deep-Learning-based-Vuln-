target_ulong helper_rdhwr_synci_step(CPUMIPSState *env)

{

    if ((env->hflags & MIPS_HFLAG_CP0) ||

        (env->CP0_HWREna & (1 << 1)))

        return env->SYNCI_Step;

    else

        do_raise_exception(env, EXCP_RI, GETPC());



    return 0;

}
