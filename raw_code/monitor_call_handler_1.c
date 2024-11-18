static void monitor_call_handler(Monitor *mon, const mon_cmd_t *cmd,

                                 const QDict *params)

{

    QObject *data = NULL;



    cmd->mhandler.cmd_new(mon, params, &data);



    if (is_async_return(data)) {

        /*

         * Asynchronous commands have no initial return data but they can

         * generate errors.  Data is returned via the async completion handler.

         */

        if (monitor_ctrl_mode(mon) && monitor_has_error(mon)) {

            monitor_protocol_emitter(mon, NULL);

        }

    } else if (monitor_ctrl_mode(mon)) {

        /* Monitor Protocol */

        monitor_protocol_emitter(mon, data);

    } else {

        /* User Protocol */

         if (data)

            cmd->user_print(mon, data);

    }



    qobject_decref(data);

}
