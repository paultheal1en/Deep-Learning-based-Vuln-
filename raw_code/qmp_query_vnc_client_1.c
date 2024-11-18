static VncClientInfo *qmp_query_vnc_client(const VncState *client)

{

    struct sockaddr_storage sa;

    socklen_t salen = sizeof(sa);

    char host[NI_MAXHOST];

    char serv[NI_MAXSERV];

    VncClientInfo *info;



    if (getpeername(client->csock, (struct sockaddr *)&sa, &salen) < 0) {

        return NULL;

    }



    if (getnameinfo((struct sockaddr *)&sa, salen,

                    host, sizeof(host),

                    serv, sizeof(serv),

                    NI_NUMERICHOST | NI_NUMERICSERV) < 0) {

        return NULL;

    }



    info = g_malloc0(sizeof(*info));

    info->base = g_malloc0(sizeof(*info->base));

    info->base->host = g_strdup(host);

    info->base->service = g_strdup(serv);

    info->base->family = inet_netfamily(sa.ss_family);

    info->base->websocket = client->websocket;



#ifdef CONFIG_VNC_TLS

    if (client->tls.session && client->tls.dname) {

        info->has_x509_dname = true;

        info->x509_dname = g_strdup(client->tls.dname);

    }

#endif

#ifdef CONFIG_VNC_SASL

    if (client->sasl.conn && client->sasl.username) {

        info->has_sasl_username = true;

        info->sasl_username = g_strdup(client->sasl.username);

    }

#endif



    return info;

}
