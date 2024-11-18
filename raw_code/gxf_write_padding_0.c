static void gxf_write_padding(ByteIOContext *pb, offset_t to_pad)

{

    while (to_pad--) {

        put_byte(pb, 0);

    }

}
