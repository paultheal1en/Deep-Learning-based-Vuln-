static ssize_t gunzip(void *dst, size_t dstlen, uint8_t *src,

                      size_t srclen)

{

    z_stream s;

    ssize_t dstbytes;

    int r, i, flags;



    /* skip header */

    i = 10;

    flags = src[3];

    if (src[2] != DEFLATED || (flags & RESERVED) != 0) {

        puts ("Error: Bad gzipped data\n");

        return -1;

    }

    if ((flags & EXTRA_FIELD) != 0)

        i = 12 + src[10] + (src[11] << 8);

    if ((flags & ORIG_NAME) != 0)

        while (src[i++] != 0)

            ;

    if ((flags & COMMENT) != 0)

        while (src[i++] != 0)

            ;

    if ((flags & HEAD_CRC) != 0)

        i += 2;

    if (i >= srclen) {

        puts ("Error: gunzip out of data in header\n");

        return -1;

    }



    s.zalloc = zalloc;

    s.zfree = zfree;



    r = inflateInit2(&s, -MAX_WBITS);

    if (r != Z_OK) {

        printf ("Error: inflateInit2() returned %d\n", r);

        return (-1);

    }

    s.next_in = src + i;

    s.avail_in = srclen - i;

    s.next_out = dst;

    s.avail_out = dstlen;

    r = inflate(&s, Z_FINISH);

    if (r != Z_OK && r != Z_STREAM_END) {

        printf ("Error: inflate() returned %d\n", r);

        return -1;

    }

    dstbytes = s.next_out - (unsigned char *) dst;

    inflateEnd(&s);



    return dstbytes;

}
