void netdev_del_completion(ReadLineState *rs, int nb_args, const char *str)

{

    int len, count, i;

    NetClientState *ncs[MAX_QUEUE_NUM];



    if (nb_args != 2) {

        return;

    }



    len = strlen(str);

    readline_set_completion_index(rs, len);

    count = qemu_find_net_clients_except(NULL, ncs, NET_CLIENT_OPTIONS_KIND_NIC,

                                         MAX_QUEUE_NUM);

    for (i = 0; i < count; i++) {

        QemuOpts *opts;

        const char *name = ncs[i]->name;

        if (strncmp(str, name, len)) {

            continue;

        }

        opts = qemu_opts_find(qemu_find_opts_err("netdev", NULL), name);

        if (opts) {

            readline_add_completion(rs, name);

        }

    }

}
