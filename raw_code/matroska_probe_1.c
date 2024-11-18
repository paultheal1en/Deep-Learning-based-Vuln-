static int matroska_probe(AVProbeData *p)

{

    uint64_t total = 0;

    int len_mask = 0x80, size = 1, n = 1, i;



    /* EBML header? */

    if (AV_RB32(p->buf) != EBML_ID_HEADER)

        return 0;



    /* length of header */

    total = p->buf[4];

    while (size <= 8 && !(total & len_mask)) {

        size++;

        len_mask >>= 1;

    }

    if (size > 8)

      return 0;

    total &= (len_mask - 1);

    while (n < size)

        total = (total << 8) | p->buf[4 + n++];



    /* Does the probe data contain the whole header? */

    if (p->buf_size < 4 + size + total)

      return 0;



    /* The header should contain a known document type. For now,

     * we don't parse the whole header but simply check for the

     * availability of that array of characters inside the header.

     * Not fully fool-proof, but good enough. */

    for (i = 0; i < FF_ARRAY_ELEMS(matroska_doctypes); i++) {

        int probelen = strlen(matroska_doctypes[i]);



        for (n = 4+size; n <= 4+size+total-probelen; n++)

            if (!memcmp(p->buf+n, matroska_doctypes[i], probelen))

                return AVPROBE_SCORE_MAX;

    }



    // probably valid EBML header but no recognized doctype

    return AVPROBE_SCORE_MAX/2;

}