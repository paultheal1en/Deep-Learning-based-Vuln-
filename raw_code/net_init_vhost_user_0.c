int net_init_vhost_user(const NetClientOptions *opts, const char *name,

                   NetClientState *peer)

{

    return net_vhost_user_init(peer, "vhost_user", 0, 0, 0);

}
