int do_migrate_cancel(Monitor *mon, const QDict *qdict, QObject **ret_data)

{

    MigrationState *s = current_migration;



    if (s)

        s->cancel(s);



    return 0;

}
