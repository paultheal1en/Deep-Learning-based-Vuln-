void migration_set_outgoing_channel(MigrationState *s,

                                    QIOChannel *ioc)

{

    QEMUFile *f = qemu_fopen_channel_output(ioc);



    s->to_dst_file = f;



    migrate_fd_connect(s);

}
