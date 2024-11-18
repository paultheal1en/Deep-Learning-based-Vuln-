static void cpu_unregister_map_client(void *_client)

{

    MapClient *client = (MapClient *)_client;



    QLIST_REMOVE(client, link);

    g_free(client);

}
