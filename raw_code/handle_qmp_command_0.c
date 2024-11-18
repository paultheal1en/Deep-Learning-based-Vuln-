static void handle_qmp_command(JSONMessageParser *parser, QList *tokens)

{

    Error *local_err = NULL;

    QObject *obj, *data;

    QDict *input, *args;

    const mon_cmd_t *cmd;

    const char *cmd_name;

    Monitor *mon = cur_mon;



    args = input = NULL;

    data = NULL;



    obj = json_parser_parse(tokens, NULL);

    if (!obj) {

        // FIXME: should be triggered in json_parser_parse()

        qerror_report(QERR_JSON_PARSING);

        goto err_out;

    }



    input = qmp_check_input_obj(obj, &local_err);

    if (!input) {

        qerror_report_err(local_err);

        qobject_decref(obj);

        goto err_out;

    }



    mon->mc->id = qdict_get(input, "id");

    qobject_incref(mon->mc->id);



    cmd_name = qdict_get_str(input, "execute");

    trace_handle_qmp_command(mon, cmd_name);

    cmd = qmp_find_cmd(cmd_name);

    if (!cmd) {

        qerror_report(ERROR_CLASS_COMMAND_NOT_FOUND,

                      "The command %s has not been found", cmd_name);

        goto err_out;

    }

    if (invalid_qmp_mode(mon, cmd)) {

        goto err_out;

    }



    obj = qdict_get(input, "arguments");

    if (!obj) {

        args = qdict_new();

    } else {

        args = qobject_to_qdict(obj);

        QINCREF(args);

    }



    qmp_check_client_args(cmd, args, &local_err);

    if (local_err) {

        qerror_report_err(local_err);

        goto err_out;

    }



    if (cmd->mhandler.cmd_new(mon, args, &data)) {

        /* Command failed... */

        if (!monitor_has_error(mon)) {

            /* ... without setting an error, so make one up */

            qerror_report(QERR_UNDEFINED_ERROR);

        }

    }



err_out:

    monitor_protocol_emitter(mon, data);

    qobject_decref(data);

    QDECREF(input);

    QDECREF(args);

}
