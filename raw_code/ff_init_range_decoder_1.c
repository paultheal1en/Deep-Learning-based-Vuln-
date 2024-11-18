av_cold void ff_init_range_decoder(RangeCoder *c, const uint8_t *buf,

                                   int buf_size)

{

    /* cast to avoid compiler warning */

    ff_init_range_encoder(c, (uint8_t *)buf, buf_size);



    c->low         = AV_RB16(c->bytestream);

    c->bytestream += 2;




