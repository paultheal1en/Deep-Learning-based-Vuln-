static void migrate_fd_monitor_suspend(MigrationState *s, Monitor *mon)

{

    if (monitor_suspend(mon) == 0) {

        DPRINTF("suspending monitor\n");

    } else {

        monitor_printf(mon, "terminal does not allow synchronous "

                       "migration, continuing detached\n");

    }

}
