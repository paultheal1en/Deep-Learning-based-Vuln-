static int vnc_zlib_stop(VncState *vs)

{

    z_streamp zstream = &vs->zlib_stream;

    int previous_out;



    // switch back to normal output/zlib buffers

    vs->zlib = vs->output;

    vs->output = vs->zlib_tmp;



    // compress the zlib buffer



    // initialize the stream

    // XXX need one stream per session

    if (zstream->opaque != vs) {

        int err;



        VNC_DEBUG("VNC: initializing zlib stream\n");

        VNC_DEBUG("VNC: opaque = %p | vs = %p\n", zstream->opaque, vs);

        zstream->zalloc = vnc_zlib_zalloc;

        zstream->zfree = vnc_zlib_zfree;



        err = deflateInit2(zstream, vs->tight_compression, Z_DEFLATED, MAX_WBITS,

                           MAX_MEM_LEVEL, Z_DEFAULT_STRATEGY);



        if (err != Z_OK) {

            fprintf(stderr, "VNC: error initializing zlib\n");

            return -1;

        }



        vs->zlib_level = vs->tight_compression;

        zstream->opaque = vs;

    }



    if (vs->tight_compression != vs->zlib_level) {

        if (deflateParams(zstream, vs->tight_compression,

                          Z_DEFAULT_STRATEGY) != Z_OK) {

            return -1;

        }

        vs->zlib_level = vs->tight_compression;

    }



    // reserve memory in output buffer

    buffer_reserve(&vs->output, vs->zlib.offset + 64);



    // set pointers

    zstream->next_in = vs->zlib.buffer;

    zstream->avail_in = vs->zlib.offset;

    zstream->next_out = vs->output.buffer + vs->output.offset;

    zstream->avail_out = vs->output.capacity - vs->output.offset;

    zstream->data_type = Z_BINARY;

    previous_out = zstream->total_out;



    // start encoding

    if (deflate(zstream, Z_SYNC_FLUSH) != Z_OK) {

        fprintf(stderr, "VNC: error during zlib compression\n");

        return -1;

    }



    vs->output.offset = vs->output.capacity - zstream->avail_out;

    return zstream->total_out - previous_out;

}
