void do_load_dcr (void)

{

    target_ulong val;



    if (unlikely(env->dcr_env == NULL)) {

        if (loglevel != 0) {

            fprintf(logfile, "No DCR environment\n");

        }

        do_raise_exception_err(EXCP_PROGRAM, EXCP_INVAL | EXCP_INVAL_INVAL);

    } else if (unlikely(ppc_dcr_read(env->dcr_env, T0, &val) != 0)) {

        if (loglevel != 0) {

            fprintf(logfile, "DCR read error %d %03x\n", (int)T0, (int)T0);

        }

        do_raise_exception_err(EXCP_PROGRAM, EXCP_INVAL | EXCP_PRIV_REG);

    } else {

        T0 = val;

    }

}
