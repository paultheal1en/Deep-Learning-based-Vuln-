char *qmp_human_monitor_command(const char *command_line, bool has_cpu_index,

                                int64_t cpu_index, Error **errp)

{

    char *output = NULL;

    Monitor *old_mon, hmp;



    memset(&hmp, 0, sizeof(hmp));

    hmp.outbuf = qstring_new();

    hmp.skip_flush = true;



    old_mon = cur_mon;

    cur_mon = &hmp;



    if (has_cpu_index) {

        int ret = monitor_set_cpu(cpu_index);

        if (ret < 0) {

            cur_mon = old_mon;

            error_set(errp, QERR_INVALID_PARAMETER_VALUE, "cpu-index",

                      "a CPU number");

            goto out;

        }

    }



    handle_user_command(&hmp, command_line);

    cur_mon = old_mon;



    if (qstring_get_length(hmp.outbuf) > 0) {

        output = g_strdup(qstring_get_str(hmp.outbuf));

    } else {

        output = g_strdup("");

    }



out:

    QDECREF(hmp.outbuf);

    return output;

}
