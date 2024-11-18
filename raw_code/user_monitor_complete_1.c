static void user_monitor_complete(void *opaque, QObject *ret_data)

{

    MonitorCompletionData *data = (MonitorCompletionData *)opaque; 



    if (ret_data) {

        data->user_print(data->mon, ret_data);

    }

    monitor_resume(data->mon);

    g_free(data);

}
