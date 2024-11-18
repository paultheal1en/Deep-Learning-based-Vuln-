static void vnc_tls_handshake_io(void *opaque) {

    struct VncState *vs = (struct VncState *)opaque;



    VNC_DEBUG("Handshake IO continue\n");

    vnc_start_vencrypt_handshake(vs);

}
