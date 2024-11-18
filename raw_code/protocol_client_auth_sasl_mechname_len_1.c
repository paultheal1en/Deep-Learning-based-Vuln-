static int protocol_client_auth_sasl_mechname_len(VncState *vs, uint8_t *data, size_t len)

{

    uint32_t mechlen = read_u32(data, 0);

    VNC_DEBUG("Got client mechname len %d\n", mechlen);

    if (mechlen > 100) {

        VNC_DEBUG("Too long SASL mechname data %d\n", mechlen);

        vnc_client_error(vs);

        return -1;

    }

    if (mechlen < 1) {

        VNC_DEBUG("Too short SASL mechname %d\n", mechlen);

        vnc_client_error(vs);

        return -1;

    }

    vnc_read_when(vs, protocol_client_auth_sasl_mechname,mechlen);

    return 0;

}
