static void add_ptimer_tests(uint8_t policy)

{

    uint8_t *ppolicy = g_malloc(1);

    char *policy_name = g_malloc0(256);



    *ppolicy = policy;



    if (policy == PTIMER_POLICY_DEFAULT) {

        g_sprintf(policy_name, "default");

    }



    if (policy & PTIMER_POLICY_WRAP_AFTER_ONE_PERIOD) {

        g_strlcat(policy_name, "wrap_after_one_period,", 256);

    }



    if (policy & PTIMER_POLICY_CONTINUOUS_TRIGGER) {

        g_strlcat(policy_name, "continuous_trigger,", 256);

    }



    if (policy & PTIMER_POLICY_NO_IMMEDIATE_TRIGGER) {

        g_strlcat(policy_name, "no_immediate_trigger,", 256);

    }



    if (policy & PTIMER_POLICY_NO_IMMEDIATE_RELOAD) {

        g_strlcat(policy_name, "no_immediate_reload,", 256);

    }



    if (policy & PTIMER_POLICY_NO_COUNTER_ROUND_DOWN) {

        g_strlcat(policy_name, "no_counter_rounddown,", 256);

    }



    g_test_add_data_func(

        g_strdup_printf("/ptimer/set_count policy=%s", policy_name),

        ppolicy, check_set_count);



    g_test_add_data_func(

        g_strdup_printf("/ptimer/set_limit policy=%s", policy_name),

        ppolicy, check_set_limit);



    g_test_add_data_func(

        g_strdup_printf("/ptimer/oneshot policy=%s", policy_name),

        ppolicy, check_oneshot);



    g_test_add_data_func(

        g_strdup_printf("/ptimer/periodic policy=%s", policy_name),

        ppolicy, check_periodic);



    g_test_add_data_func(

        g_strdup_printf("/ptimer/on_the_fly_mode_change policy=%s", policy_name),

        ppolicy, check_on_the_fly_mode_change);



    g_test_add_data_func(

        g_strdup_printf("/ptimer/on_the_fly_period_change policy=%s", policy_name),

        ppolicy, check_on_the_fly_period_change);



    g_test_add_data_func(

        g_strdup_printf("/ptimer/on_the_fly_freq_change policy=%s", policy_name),

        ppolicy, check_on_the_fly_freq_change);



    g_test_add_data_func(

        g_strdup_printf("/ptimer/run_with_period_0 policy=%s", policy_name),

        ppolicy, check_run_with_period_0);



    g_test_add_data_func(

        g_strdup_printf("/ptimer/run_with_delta_0 policy=%s", policy_name),

        ppolicy, check_run_with_delta_0);



    g_test_add_data_func(

        g_strdup_printf("/ptimer/periodic_with_load_0 policy=%s", policy_name),

        ppolicy, check_periodic_with_load_0);



    g_test_add_data_func(

        g_strdup_printf("/ptimer/oneshot_with_load_0 policy=%s", policy_name),

        ppolicy, check_oneshot_with_load_0);

}
