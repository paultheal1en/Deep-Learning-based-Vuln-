static void do_cpu_set(Monitor *mon, const QDict *qdict, QObject **ret_data)

{

    int index = qdict_get_int(qdict, "index");

    if (mon_set_cpu(index) < 0)

        qemu_error_new(QERR_INVALID_CPU_INDEX);

}
