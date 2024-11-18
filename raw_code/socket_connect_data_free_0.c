static void socket_connect_data_free(socket_connect_data *c)

{

    qapi_free_SocketAddressLegacy(c->saddr);

    g_free(c->model);

    g_free(c->name);

    g_free(c);

}
