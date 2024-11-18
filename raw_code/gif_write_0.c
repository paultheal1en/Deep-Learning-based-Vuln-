int gif_write(ByteIOContext *pb, AVImageInfo *info)

{

    gif_image_write_header(pb, info->width, info->height, 

                           (uint32_t *)info->pict.data[1]);

    gif_image_write_image(pb, 0, 0, info->width, info->height, 

                          info->pict.data[0], info->pict.linesize[0], 

                          PIX_FMT_PAL8);

    put_byte(pb, 0x3b);

    put_flush_packet(pb);

    return 0;

}
