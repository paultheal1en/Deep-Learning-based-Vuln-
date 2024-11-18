static VncServerInfo2List *qmp_query_server_entry(QIOChannelSocket *ioc,

                                                  bool websocket,

                                                  int auth,

                                                  int subauth,

                                                  VncServerInfo2List *prev)

{

    VncServerInfo2List *list;

    VncServerInfo2 *info;

    Error *err = NULL;

    SocketAddress *addr;



    addr = qio_channel_socket_get_local_address(ioc, &err);

    if (!addr) {

        error_free(err);

        return prev;

    }



    info = g_new0(VncServerInfo2, 1);

    vnc_init_basic_info(addr, qapi_VncServerInfo2_base(info), &err);

    qapi_free_SocketAddress(addr);

    if (err) {

        qapi_free_VncServerInfo2(info);

        error_free(err);

        return prev;

    }

    info->websocket = websocket;



    qmp_query_auth(auth, subauth, &info->auth,

                   &info->vencrypt, &info->has_vencrypt);



    list = g_new0(VncServerInfo2List, 1);

    list->value = info;

    list->next = prev;

    return list;

}
