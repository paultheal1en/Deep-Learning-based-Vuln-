void vnc_tight_clear(VncState *vs)

{

    int i;

    for (i=0; i<ARRAY_SIZE(vs->tight_stream); i++) {

        if (vs->tight_stream[i].opaque) {

            deflateEnd(&vs->tight_stream[i]);

        }

    }



    buffer_free(&vs->tight);

    buffer_free(&vs->tight_zlib);

    buffer_free(&vs->tight_gradient);

#ifdef CONFIG_VNC_JPEG

    buffer_free(&vs->tight_jpeg);

#endif

}
