static void net_vhost_user_event(void *opaque, int event)

{

    VhostUserState *s = opaque;



    switch (event) {

    case CHR_EVENT_OPENED:

        vhost_user_start(s);

        net_vhost_link_down(s, false);

        error_report("chardev \"%s\" went up", s->chr->label);

        break;

    case CHR_EVENT_CLOSED:

        net_vhost_link_down(s, true);

        vhost_user_stop(s);

        error_report("chardev \"%s\" went down", s->chr->label);

        break;

    }

}
