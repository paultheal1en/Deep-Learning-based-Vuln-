void qmp_disable_command(const char *name)

{

    QmpCommand *cmd;



    QTAILQ_FOREACH(cmd, &qmp_commands, node) {

        if (strcmp(cmd->name, name) == 0) {

            cmd->enabled = false;

            return;

        }

    }

}
