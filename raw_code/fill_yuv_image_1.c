static void fill_yuv_image(AVFrame *pict, int frame_index,

                           int width, int height)

{

    int x, y, i, ret;



    /* when we pass a frame to the encoder, it may keep a reference to it

     * internally;

     * make sure we do not overwrite it here

     */

    ret = av_frame_make_writable(pict);

    if (ret < 0)

        exit(1);



    i = frame_index;



    /* Y */

    for (y = 0; y < height; y++)

        for (x = 0; x < width; x++)

            pict->data[0][y * pict->linesize[0] + x] = x + y + i * 3;



    /* Cb and Cr */

    for (y = 0; y < height / 2; y++) {

        for (x = 0; x < width / 2; x++) {

            pict->data[1][y * pict->linesize[1] + x] = 128 + y + i * 2;

            pict->data[2][y * pict->linesize[2] + x] = 64 + x + i * 5;

        }

    }

}
