static bool invalid_qmp_mode(const Monitor *mon, const mon_cmd_t *cmd)

{

    bool is_cap = cmd->mhandler.cmd_new == do_qmp_capabilities;

    if (is_cap && qmp_cmd_mode(mon)) {

        qerror_report(ERROR_CLASS_COMMAND_NOT_FOUND,

                      "Capabilities negotiation is already complete, command "

                      "'%s' ignored", cmd->name);

        return true;

    }

    if (!is_cap && !qmp_cmd_mode(mon)) {

        qerror_report(ERROR_CLASS_COMMAND_NOT_FOUND,

                      "Expecting capabilities negotiation with "

                      "'qmp_capabilities' before command '%s'", cmd->name);

        return true;

    }

    return false;

}
