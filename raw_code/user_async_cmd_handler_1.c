static void user_async_cmd_handler(Monitor *mon, const mon_cmd_t *cmd,

                                   const QDict *params)

{

    int ret;



    MonitorCompletionData *cb_data = g_malloc(sizeof(*cb_data));

    cb_data->mon = mon;

    cb_data->user_print = cmd->user_print;

    monitor_suspend(mon);

    ret = cmd->mhandler.cmd_async(mon, params,

                                  user_monitor_complete, cb_data);

    if (ret < 0) {

        monitor_resume(mon);

        g_free(cb_data);

    }

}
