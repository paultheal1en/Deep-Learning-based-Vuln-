static void hmp_logfile(Monitor *mon, const QDict *qdict)

{

    qemu_set_log_filename(qdict_get_str(qdict, "filename"));

}
