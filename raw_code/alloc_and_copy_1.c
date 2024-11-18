static void alloc_and_copy(uint8_t **poutbuf,          int *poutbuf_size,

                           const uint8_t *sps_pps, uint32_t sps_pps_size,

                           const uint8_t *in,      uint32_t in_size) {

    uint32_t offset = *poutbuf_size;

    uint8_t nal_header_size = offset ? 3 : 4;



    *poutbuf_size += sps_pps_size+in_size+nal_header_size;

    *poutbuf = av_realloc(*poutbuf, *poutbuf_size);

    if (sps_pps)

        memcpy(*poutbuf+offset, sps_pps, sps_pps_size);

    memcpy(*poutbuf+sps_pps_size+nal_header_size+offset, in, in_size);

    if (!offset)

        AV_WB32(*poutbuf+sps_pps_size, 1);

    else {

        (*poutbuf+offset)[0] = (*poutbuf+offset)[1] = 0;

        (*poutbuf+offset)[2] = 1;

    }

}
