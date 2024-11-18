static void vnc_handshake_io(void *opaque) {

    struct VncState *vs = (struct VncState *)opaque;



    VNC_DEBUG("Handshake IO continue\n");

    vnc_continue_handshake(vs);

}
