ssize_t vnc_client_io_error(VncState *vs, ssize_t ret, Error **errp)

{

    if (ret <= 0) {

        if (ret == 0) {

            VNC_DEBUG("Closing down client sock: EOF\n");

        } else if (ret != QIO_CHANNEL_ERR_BLOCK) {

            VNC_DEBUG("Closing down client sock: ret %zd (%s)\n",

                      ret, errp ? error_get_pretty(*errp) : "Unknown");

        }



        vnc_disconnect_start(vs);

        if (errp) {

            error_free(*errp);

            *errp = NULL;

        }

        return 0;

    }

    return ret;

}
