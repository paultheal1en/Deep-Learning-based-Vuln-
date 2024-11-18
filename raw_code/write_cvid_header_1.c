static int write_cvid_header(CinepakEncContext *s, unsigned char *buf, int num_strips, int data_size)

{

    buf[0] = 0;

    AV_WB24(&buf[1], data_size + CVID_HEADER_SIZE);

    AV_WB16(&buf[4], s->w);

    AV_WB16(&buf[6], s->h);

    AV_WB16(&buf[8], num_strips);



    return CVID_HEADER_SIZE;

}
