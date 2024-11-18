int vncws_decode_frame(Buffer *input, uint8_t **payload,

                           size_t *payload_size, size_t *frame_size)

{

    unsigned char opcode = 0, fin = 0, has_mask = 0;

    size_t header_size = 0;

    uint32_t *payload32;

    WsHeader *header = (WsHeader *)input->buffer;

    WsMask mask;

    int i;



    if (input->offset < WS_HEAD_MIN_LEN + 4) {

        /* header not complete */

        return 0;

    }



    fin = (header->b0 & 0x80) >> 7;

    opcode = header->b0 & 0x0f;

    has_mask = (header->b1 & 0x80) >> 7;

    *payload_size = header->b1 & 0x7f;



    if (opcode == WS_OPCODE_CLOSE) {

        /* disconnect */

        return -1;

    }



    /* Websocket frame sanity check:

     * * Websocket fragmentation is not supported.

     * * All  websockets frames sent by a client have to be masked.

     * * Only binary encoding is supported.

     */

    if (!fin || !has_mask || opcode != WS_OPCODE_BINARY_FRAME) {

        VNC_DEBUG("Received faulty/unsupported Websocket frame\n");

        return -2;

    }



    if (*payload_size < 126) {

        header_size = 6;

        mask = header->u.m;

    } else if (*payload_size == 126 && input->offset >= 8) {

        *payload_size = be16_to_cpu(header->u.s16.l16);

        header_size = 8;

        mask = header->u.s16.m16;

    } else if (*payload_size == 127 && input->offset >= 14) {

        *payload_size = be64_to_cpu(header->u.s64.l64);

        header_size = 14;

        mask = header->u.s64.m64;

    } else {

        /* header not complete */

        return 0;

    }



    *frame_size = header_size + *payload_size;



    if (input->offset < *frame_size) {

        /* frame not complete */

        return 0;

    }



    *payload = input->buffer + header_size;



    /* unmask frame */

    /* process 1 frame (32 bit op) */

    payload32 = (uint32_t *)(*payload);

    for (i = 0; i < *payload_size / 4; i++) {

        payload32[i] ^= mask.u;

    }

    /* process the remaining bytes (if any) */

    for (i *= 4; i < *payload_size; i++) {

        (*payload)[i] ^= mask.c[i % 4];

    }



    return 1;

}
