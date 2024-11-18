static int gif_image_write_header(ByteIOContext *pb, 

                                  int width, int height, uint32_t *palette)

{

    int i;

    unsigned int v;



    put_tag(pb, "GIF");

    put_tag(pb, "89a");

    put_le16(pb, width);

    put_le16(pb, height);



    put_byte(pb, 0xf7); /* flags: global clut, 256 entries */

    put_byte(pb, 0x1f); /* background color index */

    put_byte(pb, 0); /* aspect ratio */



    /* the global palette */

    if (!palette) {

        put_buffer(pb, (unsigned char *)gif_clut, 216*3);

        for(i=0;i<((256-216)*3);i++)

            put_byte(pb, 0);

    } else {

        for(i=0;i<256;i++) {

            v = palette[i];

            put_byte(pb, (v >> 16) & 0xff);

            put_byte(pb, (v >> 8) & 0xff);

            put_byte(pb, (v) & 0xff);

        }

    }



    /* application extension header */

    /* XXX: not really sure what to put in here... */

#ifdef GIF_ADD_APP_HEADER

    put_byte(pb, 0x21);

    put_byte(pb, 0xff);

    put_byte(pb, 0x0b);

    put_tag(pb, "NETSCAPE2.0");

    put_byte(pb, 0x03);

    put_byte(pb, 0x01);

    put_byte(pb, 0x00);

    put_byte(pb, 0x00);

#endif

    return 0;

}
