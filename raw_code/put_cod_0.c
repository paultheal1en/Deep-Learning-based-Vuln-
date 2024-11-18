static int put_cod(Jpeg2000EncoderContext *s)

{

    Jpeg2000CodingStyle *codsty = &s->codsty;



    if (s->buf_end - s->buf < 14)

        return -1;



    bytestream_put_be16(&s->buf, JPEG2000_COD);

    bytestream_put_be16(&s->buf, 12); // Lcod

    bytestream_put_byte(&s->buf, 0);  // Scod

    // SGcod

    bytestream_put_byte(&s->buf, 0); // progression level

    bytestream_put_be16(&s->buf, 1); // num of layers

    if(s->avctx->pix_fmt == AV_PIX_FMT_YUV444P){

        bytestream_put_byte(&s->buf, 2); // ICT

    }else{

        bytestream_put_byte(&s->buf, 0); // unspecified

    }

    // SPcod

    bytestream_put_byte(&s->buf, codsty->nreslevels - 1); // num of decomp. levels

    bytestream_put_byte(&s->buf, codsty->log2_cblk_width-2); // cblk width

    bytestream_put_byte(&s->buf, codsty->log2_cblk_height-2); // cblk height

    bytestream_put_byte(&s->buf, 0); // cblk style

    bytestream_put_byte(&s->buf, codsty->transform); // transformation

    return 0;

}
