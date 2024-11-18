static void check_decode_result(int *got_output, int ret)

{

    if (*got_output || ret<0)

        decode_error_stat[ret<0] ++;



    if (ret < 0 && exit_on_error)

        exit_program(1);

}
