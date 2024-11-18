static int start_auth_vnc(VncState *vs)

{

    make_challenge(vs);

    /* Send client a 'random' challenge */

    vnc_write(vs, vs->challenge, sizeof(vs->challenge));

    vnc_flush(vs);



    vnc_read_when(vs, protocol_client_auth_vnc, sizeof(vs->challenge));

    return 0;

}
