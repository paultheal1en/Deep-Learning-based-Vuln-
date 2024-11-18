void net_cleanup(void)

{

#if !defined(_WIN32)

    VLANState *vlan;



    /* close network clients */

    for(vlan = first_vlan; vlan != NULL; vlan = vlan->next) {

        VLANClientState *vc;



        for(vc = vlan->first_client; vc != NULL; vc = vc->next) {

            if (vc->fd_read == tap_receive) {

                TAPState *s = vc->opaque;



                if (s->down_script[0])

                    launch_script(s->down_script, s->down_script_arg, s->fd);

            }

#if defined(CONFIG_VDE)

            if (vc->fd_read == vde_from_qemu) {

                VDEState *s = vc->opaque;

                vde_close(s->vde);

            }

#endif

        }

    }

#endif

}
