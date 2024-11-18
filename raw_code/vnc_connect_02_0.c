static void vnc_connect(VncDisplay *vd, QIOChannelSocket *sioc,

                        bool skipauth, bool websocket)

{

    VncState *vs = g_new0(VncState, 1);

    int i;



    vs->sioc = sioc;

    object_ref(OBJECT(vs->sioc));

    vs->ioc = QIO_CHANNEL(sioc);

    object_ref(OBJECT(vs->ioc));

    vs->vd = vd;



    buffer_init(&vs->input,          "vnc-input/%p", sioc);

    buffer_init(&vs->output,         "vnc-output/%p", sioc);

    buffer_init(&vs->jobs_buffer,    "vnc-jobs_buffer/%p", sioc);



    buffer_init(&vs->tight.tight,    "vnc-tight/%p", sioc);

    buffer_init(&vs->tight.zlib,     "vnc-tight-zlib/%p", sioc);

    buffer_init(&vs->tight.gradient, "vnc-tight-gradient/%p", sioc);

#ifdef CONFIG_VNC_JPEG

    buffer_init(&vs->tight.jpeg,     "vnc-tight-jpeg/%p", sioc);

#endif

#ifdef CONFIG_VNC_PNG

    buffer_init(&vs->tight.png,      "vnc-tight-png/%p", sioc);

#endif

    buffer_init(&vs->zlib.zlib,      "vnc-zlib/%p", sioc);

    buffer_init(&vs->zrle.zrle,      "vnc-zrle/%p", sioc);

    buffer_init(&vs->zrle.fb,        "vnc-zrle-fb/%p", sioc);

    buffer_init(&vs->zrle.zlib,      "vnc-zrle-zlib/%p", sioc);



    if (skipauth) {

	vs->auth = VNC_AUTH_NONE;

	vs->subauth = VNC_AUTH_INVALID;

    } else {

        if (websocket) {

            vs->auth = vd->ws_auth;

            vs->subauth = VNC_AUTH_INVALID;

        } else {

            vs->auth = vd->auth;

            vs->subauth = vd->subauth;

        }

    }

    VNC_DEBUG("Client sioc=%p ws=%d auth=%d subauth=%d\n",

              sioc, websocket, vs->auth, vs->subauth);



    vs->lossy_rect = g_malloc0(VNC_STAT_ROWS * sizeof (*vs->lossy_rect));

    for (i = 0; i < VNC_STAT_ROWS; ++i) {

        vs->lossy_rect[i] = g_new0(uint8_t, VNC_STAT_COLS);

    }



    VNC_DEBUG("New client on socket %p\n", vs->sioc);

    update_displaychangelistener(&vd->dcl, VNC_REFRESH_INTERVAL_BASE);

    qio_channel_set_blocking(vs->ioc, false, NULL);

    if (websocket) {

        vs->websocket = 1;

        if (vd->ws_tls) {

            vs->ioc_tag = qio_channel_add_watch(

                vs->ioc, G_IO_IN, vncws_tls_handshake_io, vs, NULL);

        } else {

            vs->ioc_tag = qio_channel_add_watch(

                vs->ioc, G_IO_IN, vncws_handshake_io, vs, NULL);

        }

    } else {

        vs->ioc_tag = qio_channel_add_watch(

            vs->ioc, G_IO_IN, vnc_client_io, vs, NULL);

    }



    vnc_client_cache_addr(vs);

    vnc_qmp_event(vs, QAPI_EVENT_VNC_CONNECTED);

    vnc_set_share_mode(vs, VNC_SHARE_MODE_CONNECTING);



    if (!vs->websocket) {

        vnc_init_state(vs);

    }



    if (vd->num_connecting > vd->connections_limit) {

        QTAILQ_FOREACH(vs, &vd->clients, next) {

            if (vs->share_mode == VNC_SHARE_MODE_CONNECTING) {

                vnc_disconnect_start(vs);

                return;

            }

        }

    }

}
