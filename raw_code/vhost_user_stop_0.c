static void vhost_user_stop(VhostUserState *s)

{

    if (vhost_user_running(s)) {

        vhost_net_cleanup(s->vhost_net);

    }



    s->vhost_net = 0;

}
