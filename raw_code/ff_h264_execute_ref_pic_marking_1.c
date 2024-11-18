int ff_h264_execute_ref_pic_marking(H264Context *h, MMCO *mmco, int mmco_count)

{

    int i, av_uninit(j);

    int current_ref_assigned = 0, err = 0;

    Picture *av_uninit(pic);



    if ((h->avctx->debug & FF_DEBUG_MMCO) && mmco_count == 0)

        av_log(h->avctx, AV_LOG_DEBUG, "no mmco here\n");



    for (i = 0; i < mmco_count; i++) {

        int av_uninit(structure), av_uninit(frame_num);

        if (h->avctx->debug & FF_DEBUG_MMCO)

            av_log(h->avctx, AV_LOG_DEBUG, "mmco:%d %d %d\n", h->mmco[i].opcode,

                   h->mmco[i].short_pic_num, h->mmco[i].long_arg);



        if (mmco[i].opcode == MMCO_SHORT2UNUSED ||

            mmco[i].opcode == MMCO_SHORT2LONG) {

            frame_num = pic_num_extract(h, mmco[i].short_pic_num, &structure);

            pic       = find_short(h, frame_num, &j);

            if (!pic) {

                if (mmco[i].opcode != MMCO_SHORT2LONG ||

                    !h->long_ref[mmco[i].long_arg]    ||

                    h->long_ref[mmco[i].long_arg]->frame_num != frame_num) {

                    av_log(h->avctx, AV_LOG_ERROR, "mmco: unref short failure\n");

                    err = AVERROR_INVALIDDATA;


                continue;





        switch (mmco[i].opcode) {

        case MMCO_SHORT2UNUSED:

            if (h->avctx->debug & FF_DEBUG_MMCO)

                av_log(h->avctx, AV_LOG_DEBUG, "mmco: unref short %d count %d\n",

                       h->mmco[i].short_pic_num, h->short_ref_count);

            remove_short(h, frame_num, structure ^ PICT_FRAME);

            break;

        case MMCO_SHORT2LONG:

                if (h->long_ref[mmco[i].long_arg] != pic)

                    remove_long(h, mmco[i].long_arg, 0);



                remove_short_at_index(h, j);

                h->long_ref[ mmco[i].long_arg ] = pic;

                if (h->long_ref[mmco[i].long_arg]) {

                    h->long_ref[mmco[i].long_arg]->long_ref = 1;

                    h->long_ref_count++;


            break;

        case MMCO_LONG2UNUSED:

            j   = pic_num_extract(h, mmco[i].long_arg, &structure);

            pic = h->long_ref[j];

            if (pic) {

                remove_long(h, j, structure ^ PICT_FRAME);

            } else if (h->avctx->debug & FF_DEBUG_MMCO)

                av_log(h->avctx, AV_LOG_DEBUG, "mmco: unref long failure\n");

            break;

        case MMCO_LONG:

                    // Comment below left from previous code as it is an interresting note.

                    /* First field in pair is in short term list or

                     * at a different long term index.

                     * This is not allowed; see 7.4.3.3, notes 2 and 3.

                     * Report the problem and keep the pair where it is,

                     * and mark this field valid.

                     */



            if (h->long_ref[mmco[i].long_arg] != h->cur_pic_ptr) {

                remove_long(h, mmco[i].long_arg, 0);






                h->long_ref[mmco[i].long_arg]           = h->cur_pic_ptr;

                h->long_ref[mmco[i].long_arg]->long_ref = 1;

                h->long_ref_count++;




            h->cur_pic_ptr->reference |= h->picture_structure;

            current_ref_assigned = 1;

            break;

        case MMCO_SET_MAX_LONG:

            assert(mmco[i].long_arg <= 16);

            // just remove the long term which index is greater than new max

            for (j = mmco[i].long_arg; j < 16; j++) {

                remove_long(h, j, 0);


            break;

        case MMCO_RESET:

            while (h->short_ref_count) {

                remove_short(h, h->short_ref[0]->frame_num, 0);


            for (j = 0; j < 16; j++) {

                remove_long(h, j, 0);


            h->frame_num  = h->cur_pic_ptr->frame_num = 0;

            h->mmco_reset = 1;

            h->cur_pic_ptr->mmco_reset = 1;

            for (j = 0; j < MAX_DELAYED_PIC_COUNT; j++)

                h->last_pocs[j] = INT_MIN;

            break;

        default: assert(0);





    if (!current_ref_assigned) {

        /* Second field of complementary field pair; the first field of

         * which is already referenced. If short referenced, it

         * should be first entry in short_ref. If not, it must exist

         * in long_ref; trying to put it on the short list here is an

         * error in the encoded bit stream (ref: 7.4.3.3, NOTE 2 and 3).

         */

        if (h->short_ref_count && h->short_ref[0] == h->cur_pic_ptr) {

            /* Just mark the second field valid */

            h->cur_pic_ptr->reference = PICT_FRAME;

        } else if (h->cur_pic_ptr->long_ref) {

            av_log(h->avctx, AV_LOG_ERROR, "illegal short term reference "

                                           "assignment for second field "

                                           "in complementary field pair "

                                           "(first field is long term)\n");

            err = AVERROR_INVALIDDATA;

        } else {

            pic = remove_short(h, h->cur_pic_ptr->frame_num, 0);

            if (pic) {

                av_log(h->avctx, AV_LOG_ERROR, "illegal short term buffer state detected\n");

                err = AVERROR_INVALIDDATA;




            if (h->short_ref_count)

                memmove(&h->short_ref[1], &h->short_ref[0],

                        h->short_ref_count * sizeof(Picture*));



            h->short_ref[0] = h->cur_pic_ptr;

            h->short_ref_count++;

            h->cur_pic_ptr->reference |= h->picture_structure;





    if (h->long_ref_count + h->short_ref_count > FFMAX(h->sps.ref_frame_count, 1)) {



        /* We have too many reference frames, probably due to corrupted

         * stream. Need to discard one frame. Prevents overrun of the

         * short_ref and long_ref buffers.

         */

        av_log(h->avctx, AV_LOG_ERROR,

               "number of reference frames (%d+%d) exceeds max (%d; probably "

               "corrupt input), discarding one\n",

               h->long_ref_count, h->short_ref_count, h->sps.ref_frame_count);

        err = AVERROR_INVALIDDATA;



        if (h->long_ref_count && !h->short_ref_count) {

            for (i = 0; i < 16; ++i)

                if (h->long_ref[i])

                    break;



            assert(i < 16);

            remove_long(h, i, 0);

        } else {

            pic = h->short_ref[h->short_ref_count - 1];

            remove_short(h, pic->frame_num, 0);





    print_short_term(h);

    print_long_term(h);



    if(err >= 0 && h->long_ref_count==0 && h->short_ref_count<=2 && h->pps.ref_count[0]<=1 + (h->picture_structure != PICT_FRAME) && h->cur_pic_ptr->f.pict_type == AV_PICTURE_TYPE_I){

        h->cur_pic_ptr->sync |= 1;

        if(!h->avctx->has_b_frames)

            h->sync = 2;




    return (h->avctx->err_recognition & AV_EF_EXPLODE) ? err : 0;
