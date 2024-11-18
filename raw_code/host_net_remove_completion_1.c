void host_net_remove_completion(ReadLineState *rs, int nb_args, const char *str)

{

    NetClientState *ncs[MAX_QUEUE_NUM];

    int count, i, len;



    len = strlen(str);

    readline_set_completion_index(rs, len);

    if (nb_args == 2) {

        count = qemu_find_net_clients_except(NULL, ncs,

                                             NET_CLIENT_OPTIONS_KIND_NONE,

                                             MAX_QUEUE_NUM);

        for (i = 0; i < count; i++) {

            int id;

            char name[16];



            if (net_hub_id_for_client(ncs[i], &id)) {

                continue;

            }

            snprintf(name, sizeof(name), "%d", id);

            if (!strncmp(str, name, len)) {

                readline_add_completion(rs, name);

            }

        }

        return;

    } else if (nb_args == 3) {

        count = qemu_find_net_clients_except(NULL, ncs,

                                             NET_CLIENT_OPTIONS_KIND_NIC,

                                             MAX_QUEUE_NUM);

        for (i = 0; i < count; i++) {

            int id;

            const char *name;



            if (ncs[i]->info->type == NET_CLIENT_OPTIONS_KIND_HUBPORT ||

                net_hub_id_for_client(ncs[i], &id)) {

                continue;

            }

            name = ncs[i]->name;

            if (!strncmp(str, name, len)) {

                readline_add_completion(rs, name);

            }

        }

        return;

    }

}
