static void combine_addr(char *buf, size_t len, const char* address,

                         uint16_t port)

{

    /* If the address-part contains a colon, it's an IPv6 IP so needs [] */

    if (strstr(address, ":")) {

        snprintf(buf, len, "[%s]:%u", address, port);

    } else {

        snprintf(buf, len, "%s:%u", address, port);

    }

}
