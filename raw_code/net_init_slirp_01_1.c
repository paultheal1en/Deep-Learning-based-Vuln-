int net_init_slirp(const Netdev *netdev, const char *name,

                   NetClientState *peer, Error **errp)

{

    /* FIXME error_setg(errp, ...) on failure */

    struct slirp_config_str *config;

    char *vnet;

    int ret;

    const NetdevUserOptions *user;

    const char **dnssearch;

    bool ipv4 = true, ipv6 = true;



    assert(netdev->type == NET_CLIENT_DRIVER_USER);

    user = &netdev->u.user;



    if ((user->has_ipv6 && user->ipv6 && !user->has_ipv4) ||

        (user->has_ipv4 && !user->ipv4)) {

        ipv4 = 0;

    }

    if ((user->has_ipv4 && user->ipv4 && !user->has_ipv6) ||

        (user->has_ipv6 && !user->ipv6)) {

        ipv6 = 0;

    }



    vnet = user->has_net ? g_strdup(user->net) :

           user->has_ip  ? g_strdup_printf("%s/24", user->ip) :

           NULL;



    dnssearch = slirp_dnssearch(user->dnssearch);



    /* all optional fields are initialized to "all bits zero" */



    net_init_slirp_configs(user->hostfwd, SLIRP_CFG_HOSTFWD);

    net_init_slirp_configs(user->guestfwd, 0);



    ret = net_slirp_init(peer, "user", name, user->q_restrict,

                         ipv4, vnet, user->host,

                         ipv6, user->ipv6_prefix, user->ipv6_prefixlen,

                         user->ipv6_host, user->hostname, user->tftp,

                         user->bootfile, user->dhcpstart,

                         user->dns, user->ipv6_dns, user->smb,

                         user->smbserver, dnssearch);



    while (slirp_configs) {

        config = slirp_configs;

        slirp_configs = config->next;

        g_free(config);

    }



    g_free(vnet);

    g_free(dnssearch);



    return ret;

}
