static void lz_unpack(const unsigned char *src, unsigned char *dest, int dest_len)

{

    const unsigned char *s;

    unsigned char *d;

    unsigned char *d_end;

    unsigned char queue[QUEUE_SIZE];

    unsigned int qpos;

    unsigned int dataleft;

    unsigned int chainofs;

    unsigned int chainlen;

    unsigned int speclen;

    unsigned char tag;

    unsigned int i, j;



    s = src;

    d = dest;

    d_end = d + dest_len;

    dataleft = AV_RL32(s);

    s += 4;

    memset(queue, 0x20, QUEUE_SIZE);

    if (AV_RL32(s) == 0x56781234) {

        s += 4;

        qpos = 0x111;

        speclen = 0xF + 3;

    } else {

        qpos = 0xFEE;

        speclen = 100;  /* no speclen */

    }



    while (dataleft > 0) {

        tag = *s++;

        if ((tag == 0xFF) && (dataleft > 8)) {

            if (d + 8 > d_end)

                return;

            for (i = 0; i < 8; i++) {

                queue[qpos++] = *d++ = *s++;

                qpos &= QUEUE_MASK;

            }

            dataleft -= 8;

        } else {

            for (i = 0; i < 8; i++) {

                if (dataleft == 0)

                    break;

                if (tag & 0x01) {

                    if (d + 1 > d_end)

                        return;

                    queue[qpos++] = *d++ = *s++;

                    qpos &= QUEUE_MASK;

                    dataleft--;

                } else {

                    chainofs = *s++;

                    chainofs |= ((*s & 0xF0) << 4);

                    chainlen = (*s++ & 0x0F) + 3;

                    if (chainlen == speclen)

                        chainlen = *s++ + 0xF + 3;

                    if (d + chainlen > d_end)

                        return;

                    for (j = 0; j < chainlen; j++) {

                        *d = queue[chainofs++ & QUEUE_MASK];

                        queue[qpos++] = *d++;

                        qpos &= QUEUE_MASK;

                    }

                    dataleft -= chainlen;

                }

                tag >>= 1;

            }

        }

    }

}
