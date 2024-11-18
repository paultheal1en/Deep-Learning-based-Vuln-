static bool invalid_qmp_mode(const Monitor *mon, const char *cmd,

                             Error **errp)

{

    bool is_cap = g_str_equal(cmd, "qmp_capabilities");



    if (is_cap && mon->qmp.in_command_mode) {

        error_set(errp, ERROR_CLASS_COMMAND_NOT_FOUND,

                  "Capabilities negotiation is already complete, command "

                  "'%s' ignored", cmd);

        return true;

    }

    if (!is_cap && !mon->qmp.in_command_mode) {

        error_set(errp, ERROR_CLASS_COMMAND_NOT_FOUND,

                  "Expecting capabilities negotiation with "

                  "'qmp_capabilities' before command '%s'", cmd);

        return true;

    }

    return false;

}
