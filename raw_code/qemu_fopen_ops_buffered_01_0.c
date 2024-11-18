QEMUFile *qemu_fopen_ops_buffered(MigrationState *migration_state)

{

    QEMUFileBuffered *s;



    s = g_malloc0(sizeof(*s));



    s->migration_state = migration_state;

    s->xfer_limit = migration_state->bandwidth_limit / 10;



    s->file = qemu_fopen_ops(s, &buffered_file_ops);



    s->timer = qemu_new_timer_ms(rt_clock, buffered_rate_tick, s);



    qemu_mod_timer(s->timer, qemu_get_clock_ms(rt_clock) + 100);



    return s->file;

}
