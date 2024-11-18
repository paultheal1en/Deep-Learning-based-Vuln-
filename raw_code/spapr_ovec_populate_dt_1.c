int spapr_ovec_populate_dt(void *fdt, int fdt_offset,

                           sPAPROptionVector *ov, const char *name)

{

    uint8_t vec[OV_MAXBYTES + 1];

    uint16_t vec_len;

    unsigned long lastbit;

    int i;



    g_assert(ov);



    lastbit = find_last_bit(ov->bitmap, OV_MAXBITS);

    /* if no bits are set, include at least 1 byte of the vector so we can

     * still encoded this in the device tree while abiding by the same

     * encoding/sizing expected in ibm,client-architecture-support

     */

    vec_len = (lastbit == OV_MAXBITS) ? 1 : lastbit / BITS_PER_BYTE + 1;

    g_assert_cmpint(vec_len, <=, OV_MAXBYTES);

    /* guest expects vector len encoded as vec_len - 1, since the length byte

     * is assumed and not included, and the first byte of the vector

     * is assumed as well

     */

    vec[0] = vec_len - 1;



    for (i = 1; i < vec_len + 1; i++) {

        vec[i] = guest_byte_from_bitmap(ov->bitmap, (i - 1) * BITS_PER_BYTE);

        if (vec[i]) {

            DPRINTFN("encoding guest vector byte %3d / %3d: 0x%.2x",

                     i, vec_len, vec[i]);

        }

    }



    return fdt_setprop(fdt, fdt_offset, name, vec, vec_len);

}
