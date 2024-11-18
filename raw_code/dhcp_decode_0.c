static void dhcp_decode(const uint8_t *buf, int size,

                        int *pmsg_type)

{

    const uint8_t *p, *p_end;

    int len, tag;



    *pmsg_type = 0;



    p = buf;

    p_end = buf + size;

    if (size < 5)

        return;

    if (memcmp(p, rfc1533_cookie, 4) != 0)

        return;

    p += 4;

    while (p < p_end) {

        tag = p[0];

        if (tag == RFC1533_PAD) {

            p++;

        } else if (tag == RFC1533_END) {

            break;

        } else {

            p++;

            if (p >= p_end)

                break;

            len = *p++;

            dprintf("dhcp: tag=0x%02x len=%d\n", tag, len);



            switch(tag) {

            case RFC2132_MSG_TYPE:

                if (len >= 1)

                    *pmsg_type = p[0];

                break;

            default:

                break;

            }

            p += len;

        }

    }

}
