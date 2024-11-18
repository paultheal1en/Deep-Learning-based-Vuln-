static void tap_cleanup(VLANClientState *nc)

{

    TAPState *s = DO_UPCAST(TAPState, nc, nc);



    if (s->vhost_net) {

        vhost_net_cleanup(s->vhost_net);


    }



    qemu_purge_queued_packets(nc);



    if (s->down_script[0])

        launch_script(s->down_script, s->down_script_arg, s->fd);



    tap_read_poll(s, 0);

    tap_write_poll(s, 0);

    close(s->fd);

}