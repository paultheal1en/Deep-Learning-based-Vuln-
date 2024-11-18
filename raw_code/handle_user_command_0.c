static void handle_user_command(Monitor *mon, const char *cmdline)

{

    QDict *qdict;

    const mon_cmd_t *cmd;



    qdict = qdict_new();



    cmd = monitor_parse_command(mon, cmdline, qdict);

    if (!cmd)

        goto out;



    if (monitor_handler_is_async(cmd)) {

        user_async_cmd_handler(mon, cmd, qdict);

    } else if (monitor_handler_ported(cmd)) {

        monitor_call_handler(mon, cmd, qdict);

    } else {

        cmd->mhandler.cmd(mon, qdict);

    }



    if (monitor_has_error(mon))

        monitor_print_error(mon);



out:

    QDECREF(qdict);

}
