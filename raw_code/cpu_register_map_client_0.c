void *cpu_register_map_client(void *opaque, void (*callback)(void *opaque))

{

    MapClient *client = qemu_malloc(sizeof(*client));



    client->opaque = opaque;

    client->callback = callback;

    LIST_INSERT_HEAD(&map_client_list, client, link);

    return client;

}
