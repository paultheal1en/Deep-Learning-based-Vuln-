int net_init_hubport(const NetClientOptions *opts, const char *name,

                     NetClientState *peer)

{

    const NetdevHubPortOptions *hubport;



    assert(opts->kind == NET_CLIENT_OPTIONS_KIND_HUBPORT);

    hubport = opts->hubport;



    if (peer) {

        return -EINVAL;

    }



    net_hub_add_port(hubport->hubid, name);

    return 0;

}
