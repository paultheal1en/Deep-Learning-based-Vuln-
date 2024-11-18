static int qmp_check_client_args(const mon_cmd_t *cmd, QDict *client_args)

{

    int flags, err;

    QDict *cmd_args;



    cmd_args = qdict_from_args_type(cmd->args_type);



    flags = 0;

    err = check_mandatory_args(cmd_args, client_args, &flags);

    if (err) {

        goto out;

    }



    /* TODO: Check client args type */



out:

    QDECREF(cmd_args);

    return err;

}
