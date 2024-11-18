static void core_rtas_register_types(void)

{

    spapr_rtas_register(RTAS_DISPLAY_CHARACTER, "display-character",

                        rtas_display_character);

    spapr_rtas_register(RTAS_GET_TIME_OF_DAY, "get-time-of-day",

                        rtas_get_time_of_day);

    spapr_rtas_register(RTAS_SET_TIME_OF_DAY, "set-time-of-day",

                        rtas_set_time_of_day);

    spapr_rtas_register(RTAS_POWER_OFF, "power-off", rtas_power_off);

    spapr_rtas_register(RTAS_SYSTEM_REBOOT, "system-reboot",

                        rtas_system_reboot);

    spapr_rtas_register(RTAS_QUERY_CPU_STOPPED_STATE, "query-cpu-stopped-state",

                        rtas_query_cpu_stopped_state);

    spapr_rtas_register(RTAS_START_CPU, "start-cpu", rtas_start_cpu);

    spapr_rtas_register(RTAS_STOP_SELF, "stop-self", rtas_stop_self);

    spapr_rtas_register(RTAS_IBM_GET_SYSTEM_PARAMETER,

                        "ibm,get-system-parameter",

                        rtas_ibm_get_system_parameter);

    spapr_rtas_register(RTAS_IBM_SET_SYSTEM_PARAMETER,

                        "ibm,set-system-parameter",

                        rtas_ibm_set_system_parameter);



}