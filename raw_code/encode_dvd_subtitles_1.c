static int encode_dvd_subtitles(AVCodecContext *avctx,

                                uint8_t *outbuf, int outbuf_size,

                                const AVSubtitle *h)

{

    DVDSubtitleContext *dvdc = avctx->priv_data;

    uint8_t *q, *qq;

    int offset1, offset2;

    int i, rects = h->num_rects, ret;

    unsigned global_palette_hits[33] = { 0 };

    int cmap[256];

    int out_palette[4];

    int out_alpha[4];

    AVSubtitleRect vrect;

    uint8_t *vrect_data = NULL;

    int x2, y2;



    if (rects == 0 || h->rects == NULL)







    vrect = *h->rects[0];



    if (rects > 1) {

        /* DVD subtitles can have only one rectangle: build a virtual

           rectangle containing all actual rectangles.

           The data of the rectangles will be copied later, when the palette

           is decided, because the rectangles may have different palettes. */

        int xmin = h->rects[0]->x, xmax = xmin + h->rects[0]->w;

        int ymin = h->rects[0]->y, ymax = ymin + h->rects[0]->h;

        for (i = 1; i < rects; i++) {

            xmin = FFMIN(xmin, h->rects[i]->x);

            ymin = FFMIN(ymin, h->rects[i]->y);

            xmax = FFMAX(xmax, h->rects[i]->x + h->rects[i]->w);

            ymax = FFMAX(ymax, h->rects[i]->y + h->rects[i]->h);


        vrect.x = xmin;

        vrect.y = ymin;

        vrect.w = xmax - xmin;

        vrect.h = ymax - ymin;

        if ((ret = av_image_check_size(vrect.w, vrect.h, 0, avctx)) < 0)

            return ret;



        /* Count pixels outside the virtual rectangle as transparent */

        global_palette_hits[0] = vrect.w * vrect.h;


            global_palette_hits[0] -= h->rects[i]->w * h->rects[i]->h;





        count_colors(avctx, global_palette_hits, h->rects[i]);

    select_palette(avctx, out_palette, out_alpha, global_palette_hits);



    if (rects > 1) {

        if (!(vrect_data = av_calloc(vrect.w, vrect.h)))

            return AVERROR(ENOMEM);

        vrect.pict.data    [0] = vrect_data;

        vrect.pict.linesize[0] = vrect.w;

        for (i = 0; i < rects; i++) {

            build_color_map(avctx, cmap, (uint32_t *)h->rects[i]->pict.data[1],

                            out_palette, out_alpha);

            copy_rectangle(&vrect, h->rects[i], cmap);


        for (i = 0; i < 4; i++)

            cmap[i] = i;

    } else {

        build_color_map(avctx, cmap, (uint32_t *)h->rects[0]->pict.data[1],

                        out_palette, out_alpha);




    av_log(avctx, AV_LOG_DEBUG, "Selected palette:");

    for (i = 0; i < 4; i++)

        av_log(avctx, AV_LOG_DEBUG, " 0x%06x@@%02x (0x%x,0x%x)",

               dvdc->global_palette[out_palette[i]], out_alpha[i],

               out_palette[i], out_alpha[i] >> 4);

    av_log(avctx, AV_LOG_DEBUG, "\n");



    // encode data block

    q = outbuf + 4;

    offset1 = q - outbuf;

    // worst case memory requirement: 1 nibble per pixel..

    if ((q - outbuf) + vrect.w * vrect.h / 2 + 17 + 21 > outbuf_size) {

        av_log(NULL, AV_LOG_ERROR, "dvd_subtitle too big\n");

        ret = AVERROR_BUFFER_TOO_SMALL;

        goto fail;


    dvd_encode_rle(&q, vrect.pict.data[0], vrect.w * 2,

                   vrect.w, (vrect.h + 1) >> 1, cmap);

    offset2 = q - outbuf;

    dvd_encode_rle(&q, vrect.pict.data[0] + vrect.w, vrect.w * 2,

                   vrect.w, vrect.h >> 1, cmap);



    // set data packet size

    qq = outbuf + 2;

    bytestream_put_be16(&qq, q - outbuf);



    // send start display command

    bytestream_put_be16(&q, (h->start_display_time*90) >> 10);

    bytestream_put_be16(&q, (q - outbuf) /*- 2 */ + 8 + 12 + 2);

    *q++ = 0x03; // palette - 4 nibbles

    *q++ = (out_palette[3] << 4) | out_palette[2];

    *q++ = (out_palette[1] << 4) | out_palette[0];

    *q++ = 0x04; // alpha - 4 nibbles

    *q++ = (out_alpha[3] & 0xF0) | (out_alpha[2] >> 4);

    *q++ = (out_alpha[1] & 0xF0) | (out_alpha[0] >> 4);



    // 12 bytes per rect

    x2 = vrect.x + vrect.w - 1;

    y2 = vrect.y + vrect.h - 1;



    *q++ = 0x05;

    // x1 x2 -> 6 nibbles

    *q++ = vrect.x >> 4;

    *q++ = (vrect.x << 4) | ((x2 >> 8) & 0xf);

    *q++ = x2;

    // y1 y2 -> 6 nibbles

    *q++ = vrect.y >> 4;

    *q++ = (vrect.y << 4) | ((y2 >> 8) & 0xf);

    *q++ = y2;



    *q++ = 0x06;

    // offset1, offset2

    bytestream_put_be16(&q, offset1);

    bytestream_put_be16(&q, offset2);



    *q++ = 0x01; // start command

    *q++ = 0xff; // terminating command



    // send stop display command last

    bytestream_put_be16(&q, (h->end_display_time*90) >> 10);

    bytestream_put_be16(&q, (q - outbuf) - 2 /*+ 4*/);

    *q++ = 0x02; // set end

    *q++ = 0xff; // terminating command



    qq = outbuf;

    bytestream_put_be16(&qq, q - outbuf);



    av_log(NULL, AV_LOG_DEBUG, "subtitle_packet size=%td\n", q - outbuf);

    ret = q - outbuf;



fail:

    av_free(vrect_data);

    return ret;
