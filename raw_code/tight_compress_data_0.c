static int tight_compress_data(VncState *vs, int stream_id, size_t bytes,

                               int level, int strategy)

{

    z_streamp zstream = &vs->tight_stream[stream_id];

    int previous_out;



    if (bytes < VNC_TIGHT_MIN_TO_COMPRESS) {

        vnc_write(vs, vs->tight.buffer, vs->tight.offset);

        return bytes;

    }



    if (tight_init_stream(vs, stream_id, level, strategy)) {

        return -1;

    }



    /* reserve memory in output buffer */

    buffer_reserve(&vs->tight_zlib, bytes + 64);



    /* set pointers */

    zstream->next_in = vs->tight.buffer;

    zstream->avail_in = vs->tight.offset;

    zstream->next_out = vs->tight_zlib.buffer + vs->tight_zlib.offset;

    zstream->avail_out = vs->tight_zlib.capacity - vs->tight_zlib.offset;

    zstream->data_type = Z_BINARY;

    previous_out = zstream->total_out;



    /* start encoding */

    if (deflate(zstream, Z_SYNC_FLUSH) != Z_OK) {

        fprintf(stderr, "VNC: error during tight compression\n");

        return -1;

    }



    vs->tight_zlib.offset = vs->tight_zlib.capacity - zstream->avail_out;

    bytes = zstream->total_out - previous_out;



    tight_send_compact_size(vs, bytes);

    vnc_write(vs, vs->tight_zlib.buffer, bytes);



    buffer_reset(&vs->tight_zlib);



    return bytes;

}
