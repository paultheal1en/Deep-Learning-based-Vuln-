static void do_info(Monitor *mon, const QDict *qdict, QObject **ret_data)

{

    const mon_cmd_t *cmd;

    const char *item = qdict_get_try_str(qdict, "item");



    if (!item)

        goto help;



    for (cmd = info_cmds; cmd->name != NULL; cmd++) {

        if (compare_cmd(item, cmd->name))

            break;

    }



    if (cmd->name == NULL)

        goto help;



    if (monitor_handler_ported(cmd)) {

        cmd->mhandler.info_new(mon, ret_data);

        if (*ret_data)

            cmd->user_print(mon, *ret_data);

    } else {

        cmd->mhandler.info(mon);

    }



    return;



help:

    help_cmd(mon, "info");

}
