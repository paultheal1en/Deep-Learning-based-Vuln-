int ff_mjpeg_decode_sos(MJpegDecodeContext *s)

{

    int len, nb_components, i, h, v, predictor, point_transform;

    int index, id;

    const int block_size= s->lossless ? 1 : 8;

    int ilv, prev_shift;



    /* XXX: verify len field validity */

    len = get_bits(&s->gb, 16);

    nb_components = get_bits(&s->gb, 8);





    if (len != 6+2*nb_components)

    {

        av_log(s->avctx, AV_LOG_ERROR, "decode_sos: invalid len (%d)\n", len);



    for(i=0;i<nb_components;i++) {

        id = get_bits(&s->gb, 8) - 1;

        av_log(s->avctx, AV_LOG_DEBUG, "component: %d\n", id);

        /* find component index */

        for(index=0;index<s->nb_components;index++)

            if (id == s->component_id[index])

                break;

        if (index == s->nb_components)

        {

            av_log(s->avctx, AV_LOG_ERROR, "decode_sos: index(%d) out of components\n", index);





        s->comp_index[i] = index;



        s->nb_blocks[i] = s->h_count[index] * s->v_count[index];

        s->h_scount[i] = s->h_count[index];

        s->v_scount[i] = s->v_count[index];



        s->dc_index[i] = get_bits(&s->gb, 4);

        s->ac_index[i] = get_bits(&s->gb, 4);



        if (s->dc_index[i] <  0 || s->ac_index[i] < 0 ||

            s->dc_index[i] >= 4 || s->ac_index[i] >= 4)

            goto out_of_range;

        if (!s->vlcs[0][s->dc_index[i]].table || !s->vlcs[1][s->ac_index[i]].table)

            goto out_of_range;




    predictor= get_bits(&s->gb, 8); /* JPEG Ss / lossless JPEG predictor /JPEG-LS NEAR */

    ilv= get_bits(&s->gb, 8);    /* JPEG Se / JPEG-LS ILV */

    prev_shift = get_bits(&s->gb, 4); /* Ah */

    point_transform= get_bits(&s->gb, 4); /* Al */



    for(i=0;i<nb_components;i++)

        s->last_dc[i] = 1024;



    if (nb_components > 1) {

        /* interleaved stream */

        s->mb_width  = (s->width  + s->h_max * block_size - 1) / (s->h_max * block_size);

        s->mb_height = (s->height + s->v_max * block_size - 1) / (s->v_max * block_size);

    } else if(!s->ls) { /* skip this for JPEG-LS */

        h = s->h_max / s->h_scount[0];

        v = s->v_max / s->v_scount[0];

        s->mb_width  = (s->width  + h * block_size - 1) / (h * block_size);

        s->mb_height = (s->height + v * block_size - 1) / (v * block_size);

        s->nb_blocks[0] = 1;

        s->h_scount[0] = 1;

        s->v_scount[0] = 1;




    if(s->avctx->debug & FF_DEBUG_PICT_INFO)

        av_log(s->avctx, AV_LOG_DEBUG, "%s %s p:%d >>:%d ilv:%d bits:%d %s\n", s->lossless ? "lossless" : "sequencial DCT", s->rgb ? "RGB" : "",

               predictor, point_transform, ilv, s->bits,

               s->pegasus_rct ? "PRCT" : (s->rct ? "RCT" : ""));





    /* mjpeg-b can have padding bytes between sos and image data, skip them */

    for (i = s->mjpb_skiptosod; i > 0; i--)

        skip_bits(&s->gb, 8);



    if(s->lossless){

        if(CONFIG_JPEGLS_DECODER && s->ls){

//            for(){

//            reset_ls_coding_parameters(s, 0);



            if(ff_jpegls_decode_picture(s, predictor, point_transform, ilv) < 0)


        }else{

            if(s->rgb){

                if(ljpeg_decode_rgb_scan(s, predictor, point_transform) < 0)


            }else{

                if(ljpeg_decode_yuv_scan(s, predictor, point_transform) < 0)




    }else{

        if(s->progressive && predictor) {

            if(mjpeg_decode_scan_progressive_ac(s, predictor, ilv, prev_shift, point_transform) < 0)


        } else {

            if(mjpeg_decode_scan(s, nb_components, prev_shift, point_transform) < 0)




    emms_c();

    return 0;

 out_of_range:

    av_log(s->avctx, AV_LOG_ERROR, "decode_sos: ac/dc index out of range\n");

