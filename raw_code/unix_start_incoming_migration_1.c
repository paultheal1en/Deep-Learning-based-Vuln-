void unix_start_incoming_migration(const char *path, Error **errp)

{

    int s;



    s = unix_listen(path, NULL, 0, errp);

    if (s < 0) {

        return;

    }



    qemu_set_fd_handler2(s, NULL, unix_accept_incoming_migration, NULL,

                         (void *)(intptr_t)s);

}
