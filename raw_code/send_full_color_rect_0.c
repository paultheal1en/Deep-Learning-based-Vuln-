static int send_full_color_rect(VncState *vs, int w, int h)

{

    int stream = 0;

    size_t bytes;



    vnc_write_u8(vs, stream << 4); /* no flushing, no filter */



    if (vs->tight_pixel24) {

        tight_pack24(vs, vs->tight.buffer, w * h, &vs->tight.offset);

        bytes = 3;

    } else {

        bytes = vs->clientds.pf.bytes_per_pixel;

    }



    bytes = tight_compress_data(vs, stream, w * h * bytes,

                                tight_conf[vs->tight_compression].raw_zlib_level,

                                Z_DEFAULT_STRATEGY);



    return (bytes >= 0);

}
