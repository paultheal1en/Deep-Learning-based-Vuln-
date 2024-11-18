static void mxf_read_pixel_layout(ByteIOContext *pb, MXFDescriptor *descriptor)

{

    int code, value, ofs = 0;

    char layout[16] = {};



    do {

        code = get_byte(pb);

        value = get_byte(pb);

        dprintf(NULL, "pixel layout: code %#x\n", code);



        if (ofs < 16) {

            layout[ofs++] = code;

            layout[ofs++] = value;

        }

    } while (code != 0); /* SMPTE 377M E.2.46 */



    ff_mxf_decode_pixel_layout(layout, &descriptor->pix_fmt);

}
