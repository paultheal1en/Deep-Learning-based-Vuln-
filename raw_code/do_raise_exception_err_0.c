void do_raise_exception_err (uint32_t exception, int error_code)

{

#if 0

    printf("Raise exception %3x code : %d\n", exception, error_code);

#endif

    switch (exception) {

    case EXCP_PROGRAM:

        if (error_code == EXCP_FP && msr_fe0 == 0 && msr_fe1 == 0)

            return;

        break;

    default:

        break;

    }

    env->exception_index = exception;

    env->error_code = error_code;

    cpu_loop_exit();

}
