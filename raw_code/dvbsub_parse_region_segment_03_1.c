static void dvbsub_parse_region_segment(AVCodecContext *avctx,

                                        uint8_t *buf, int buf_size)

{

    DVBSubContext *ctx = (DVBSubContext*) avctx->priv_data;



    uint8_t *buf_end = buf + buf_size;

    int region_id, object_id;

    DVBSubRegion *region;

    DVBSubObject *object;

    DVBSubObjectDisplay *display;

    int fill;



    if (buf_size < 10)

        return;



    region_id = *buf++;



    region = get_region(ctx, region_id);



    if (region == NULL)

    {

        region = av_mallocz(sizeof(DVBSubRegion));



        region->id = region_id;



        region->next = ctx->region_list;

        ctx->region_list = region;




    fill = ((*buf++) >> 3) & 1;



    region->width = AV_RB16(buf);

    buf += 2;

    region->height = AV_RB16(buf);

    buf += 2;



    if (region->width * region->height != region->buf_size) {

        if (region->pbuf != 0)

            av_free(region->pbuf);



        region->buf_size = region->width * region->height;



        region->pbuf = av_malloc(region->buf_size);



        fill = 1;




    region->depth = 1 << (((*buf++) >> 2) & 7);





    region->clut = *buf++;



    if (region->depth == 8)

        region->bgcolour = *buf++;

    else {

        buf += 1;



        if (region->depth == 4)

            region->bgcolour = (((*buf++) >> 4) & 15);

        else

            region->bgcolour = (((*buf++) >> 2) & 3);




#ifdef DEBUG

    av_log(avctx, AV_LOG_INFO, "Region %d, (%dx%d)\n", region_id, region->width, region->height);

#endif



    if (fill) {

        memset(region->pbuf, region->bgcolour, region->buf_size);

#ifdef DEBUG

        av_log(avctx, AV_LOG_INFO, "Fill region (%d)\n", region->bgcolour);

#endif




    delete_region_display_list(ctx, region);



    while (buf + 5 < buf_end) {

        object_id = AV_RB16(buf);

        buf += 2;



        object = get_object(ctx, object_id);



        if (object == NULL) {

            object = av_mallocz(sizeof(DVBSubObject));



            object->id = object_id;

            object->next = ctx->object_list;

            ctx->object_list = object;




        object->type = (*buf) >> 6;



        display = av_mallocz(sizeof(DVBSubObjectDisplay));



        display->object_id = object_id;

        display->region_id = region_id;



        display->x_pos = AV_RB16(buf) & 0xfff;

        buf += 2;

        display->y_pos = AV_RB16(buf) & 0xfff;

        buf += 2;



        if ((object->type == 1 || object->type == 2) && buf+1 < buf_end) {

            display->fgcolour = *buf++;

            display->bgcolour = *buf++;




        display->region_list_next = region->display_list;

        region->display_list = display;



        display->object_list_next = object->display_list;

        object->display_list = display;

