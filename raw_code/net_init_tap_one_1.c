static void net_init_tap_one(const NetdevTapOptions *tap, NetClientState *peer,

                             const char *model, const char *name,

                             const char *ifname, const char *script,

                             const char *downscript, const char *vhostfdname,

                             int vnet_hdr, int fd, Error **errp)

{

    Error *err = NULL;

    TAPState *s = net_tap_fd_init(peer, model, name, fd, vnet_hdr);

    int vhostfd;



    tap_set_sndbuf(s->fd, tap, &err);

    if (err) {

        error_propagate(errp, err);

        return;

    }



    if (tap->has_fd || tap->has_fds) {

        snprintf(s->nc.info_str, sizeof(s->nc.info_str), "fd=%d", fd);

    } else if (tap->has_helper) {

        snprintf(s->nc.info_str, sizeof(s->nc.info_str), "helper=%s",

                 tap->helper);

    } else {

        snprintf(s->nc.info_str, sizeof(s->nc.info_str),

                 "ifname=%s,script=%s,downscript=%s", ifname, script,

                 downscript);



        if (strcmp(downscript, "no") != 0) {

            snprintf(s->down_script, sizeof(s->down_script), "%s", downscript);

            snprintf(s->down_script_arg, sizeof(s->down_script_arg),

                     "%s", ifname);

        }

    }



    if (tap->has_vhost ? tap->vhost :

        vhostfdname || (tap->has_vhostforce && tap->vhostforce)) {

        VhostNetOptions options;



        options.backend_type = VHOST_BACKEND_TYPE_KERNEL;

        options.net_backend = &s->nc;

        options.force = tap->has_vhostforce && tap->vhostforce;



        if (tap->has_vhostfd || tap->has_vhostfds) {

            vhostfd = monitor_fd_param(cur_mon, vhostfdname, &err);

            if (vhostfd == -1) {

                error_propagate(errp, err);

                return;

            }

        } else {

            vhostfd = open("/dev/vhost-net", O_RDWR);

            if (vhostfd < 0) {

                error_setg_errno(errp, errno,

                                 "tap: open vhost char device failed");

                return;

            }

        }

        options.opaque = (void *)(uintptr_t)vhostfd;



        s->vhost_net = vhost_net_init(&options);

        if (!s->vhost_net) {

            error_setg(errp,

                       "vhost-net requested but could not be initialized");

            return;

        }

    } else if (tap->has_vhostfd || tap->has_vhostfds) {

        error_setg(errp, "vhostfd= is not valid without vhost");

    }

}
