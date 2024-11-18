static void vncws_tls_handshake_io(void *opaque)

{

    struct VncState *vs = (struct VncState *)opaque;



    VNC_DEBUG("Handshake IO continue\n");

    vncws_start_tls_handshake(vs);

}
