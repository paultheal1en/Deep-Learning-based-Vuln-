void op_cp1_enabled(void)

{

    if (!(env->CP0_Status & (1 << CP0St_CU1))) {

        CALL_FROM_TB2(do_raise_exception_err, EXCP_CpU, 1);

    }

    RETURN();

}
