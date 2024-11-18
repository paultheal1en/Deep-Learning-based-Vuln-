static void handler_audit(Monitor *mon, const mon_cmd_t *cmd, int ret)

{

    if (ret && !monitor_has_error(mon)) {

        /*

         * If it returns failure, it must have passed on error.

         *

         * Action: Report an internal error to the client if in QMP.

         */

        if (monitor_ctrl_mode(mon)) {

            qerror_report(QERR_UNDEFINED_ERROR);

        }

        MON_DEBUG("command '%s' returned failure but did not pass an error\n",

                  cmd->name);

    }



#ifdef CONFIG_DEBUG_MONITOR

    if (!ret && monitor_has_error(mon)) {

        /*

         * If it returns success, it must not have passed an error.

         *

         * Action: Report the passed error to the client.

         */

        MON_DEBUG("command '%s' returned success but passed an error\n",

                  cmd->name);

    }



    if (mon_print_count_get(mon) > 0 && strcmp(cmd->name, "info") != 0) {

        /*

         * Handlers should not call Monitor print functions.

         *

         * Action: Ignore them in QMP.

         *

         * (XXX: we don't check any 'info' or 'query' command here

         * because the user print function _is_ called by do_info(), hence

         * we will trigger this check. This problem will go away when we

         * make 'query' commands real and kill do_info())

         */

        MON_DEBUG("command '%s' called print functions %d time(s)\n",

                  cmd->name, mon_print_count_get(mon));

    }

#endif

}
