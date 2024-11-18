static void check_cpu_flag(const char *name, int flag)

{

    int old_cpu_flag = state.cpu_flag;



    flag |= old_cpu_flag;

    av_set_cpu_flags_mask(flag);

    state.cpu_flag = av_get_cpu_flags();



    if (!flag || state.cpu_flag != old_cpu_flag) {

        int i;



        state.cpu_flag_name = name;

        for (i = 0; tests[i].func; i++) {

            state.current_test_name = tests[i].name;

            tests[i].func();

        }

    }

}
