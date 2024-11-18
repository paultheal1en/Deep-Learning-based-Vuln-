int do_migrate_set_speed(Monitor *mon, const QDict *qdict, QObject **ret_data)

{

    int64_t d;

    FdMigrationState *s;



    d = qdict_get_int(qdict, "value");

    d = MAX(0, MIN(UINT32_MAX, d));

    max_throttle = d;



    s = migrate_to_fms(current_migration);

    if (s && s->file) {

        qemu_file_set_rate_limit(s->file, max_throttle);

    }



    return 0;

}
