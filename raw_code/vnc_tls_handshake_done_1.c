static void vnc_tls_handshake_done(QIOTask *task,

                                   gpointer user_data)

{

    VncState *vs = user_data;

    Error *err = NULL;



    if (qio_task_propagate_error(task, &err)) {

        VNC_DEBUG("Handshake failed %s\n",

                  error_get_pretty(err));

        vnc_client_error(vs);

        error_free(err);

    } else {

        vs->ioc_tag = qio_channel_add_watch(

            vs->ioc, G_IO_IN | G_IO_OUT, vnc_client_io, vs, NULL);

        start_auth_vencrypt_subauth(vs);

    }

}
