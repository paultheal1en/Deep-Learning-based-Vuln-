void qerror_report(const char *fmt, ...)

{

    va_list va;

    QError *qerror;



    va_start(va, fmt);

    qerror = qerror_from_info(fmt, &va);

    va_end(va);



    if (monitor_cur_is_qmp()) {

        monitor_set_error(cur_mon, qerror);

    } else {

        qerror_print(qerror);

        QDECREF(qerror);

    }

}
