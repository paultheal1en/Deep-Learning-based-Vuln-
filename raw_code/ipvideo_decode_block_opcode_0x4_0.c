static int ipvideo_decode_block_opcode_0x4(IpvideoContext *s)

{

    int x, y;

    unsigned char B, BL, BH;



    /* copy a block from the previous frame; need 1 more byte */

    CHECK_STREAM_PTR(1);



    B = *s->stream_ptr++;

    BL = B & 0x0F;

    BH = (B >> 4) & 0x0F;

    x = -8 + BL;

    y = -8 + BH;



    debug_interplay ("    motion byte = %d, (x, y) = (%d, %d)\n", B, x, y);

    return copy_from(s, &s->last_frame, x, y);

}
