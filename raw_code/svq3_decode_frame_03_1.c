static int svq3_decode_frame (AVCodecContext *avctx,

                              void *data, int *data_size,

                              uint8_t *buf, int buf_size) {

  MpegEncContext *const s = avctx->priv_data;

  H264Context *const h = avctx->priv_data;

  int m, mb_type;

  unsigned char *extradata;

  unsigned int size;



  s->flags = avctx->flags;

  s->flags2 = avctx->flags2;

  s->unrestricted_mv = 1;



  if (!s->context_initialized) {

    s->width = avctx->width;

    s->height = avctx->height;

    h->pred4x4[DIAG_DOWN_LEFT_PRED] = pred4x4_down_left_svq3_c;

    h->pred16x16[PLANE_PRED8x8] = pred16x16_plane_svq3_c;

    h->halfpel_flag = 1;

    h->thirdpel_flag = 1;

    h->unknown_svq3_flag = 0;

    h->chroma_qp = 4;



    if (MPV_common_init (s) < 0)

      return -1;



    h->b_stride = 4*s->mb_width;



    alloc_tables (h);



    /* prowl for the "SEQH" marker in the extradata */

    extradata = (unsigned char *)avctx->extradata;

    for (m = 0; m < avctx->extradata_size; m++) {

      if (!memcmp (extradata, "SEQH", 4))

        break;

      extradata++;

    }



    /* if a match was found, parse the extra data */

    if (extradata && !memcmp (extradata, "SEQH", 4)) {



      GetBitContext gb;



      size = AV_RB32(&extradata[4]);

      init_get_bits (&gb, extradata + 8, size*8);



      /* 'frame size code' and optional 'width, height' */

      if (get_bits (&gb, 3) == 7) {

        get_bits (&gb, 12);

        get_bits (&gb, 12);

      }



      h->halfpel_flag = get_bits1 (&gb);

      h->thirdpel_flag = get_bits1 (&gb);



      /* unknown fields */

      get_bits1 (&gb);

      get_bits1 (&gb);

      get_bits1 (&gb);

      get_bits1 (&gb);



      s->low_delay = get_bits1 (&gb);



      /* unknown field */

      get_bits1 (&gb);



      while (get_bits1 (&gb)) {

        get_bits (&gb, 8);

      }



      h->unknown_svq3_flag = get_bits1 (&gb);

      avctx->has_b_frames = !s->low_delay;

    }

  }



  /* special case for last picture */

  if (buf_size == 0) {

    if (s->next_picture_ptr && !s->low_delay) {

      *(AVFrame *) data = *(AVFrame *) &s->next_picture;

      *data_size = sizeof(AVFrame);

    }

    return 0;

  }



  init_get_bits (&s->gb, buf, 8*buf_size);



  s->mb_x = s->mb_y = 0;



  if (svq3_decode_slice_header (h))

    return -1;



  s->pict_type = h->slice_type;

  s->picture_number = h->slice_num;



  if(avctx->debug&FF_DEBUG_PICT_INFO){

      av_log(h->s.avctx, AV_LOG_DEBUG, "%c hpel:%d, tpel:%d aqp:%d qp:%d\n",

      av_get_pict_type_char(s->pict_type), h->halfpel_flag, h->thirdpel_flag,

      s->adaptive_quant, s->qscale

      );

  }



  /* for hurry_up==5 */

  s->current_picture.pict_type = s->pict_type;

  s->current_picture.key_frame = (s->pict_type == I_TYPE);



  /* skip b frames if we dont have reference frames */

  if (s->last_picture_ptr == NULL && s->pict_type == B_TYPE) return 0;

  /* skip b frames if we are in a hurry */

  if (avctx->hurry_up && s->pict_type == B_TYPE) return 0;

  /* skip everything if we are in a hurry >= 5 */

  if (avctx->hurry_up >= 5) return 0;

  if(  (avctx->skip_frame >= AVDISCARD_NONREF && s->pict_type==B_TYPE)

     ||(avctx->skip_frame >= AVDISCARD_NONKEY && s->pict_type!=I_TYPE)

     || avctx->skip_frame >= AVDISCARD_ALL)

      return 0;



  if (s->next_p_frame_damaged) {

    if (s->pict_type == B_TYPE)

      return 0;

    else

      s->next_p_frame_damaged = 0;

  }



  frame_start (h);



  if (s->pict_type == B_TYPE) {

    h->frame_num_offset = (h->slice_num - h->prev_frame_num);



    if (h->frame_num_offset < 0) {

      h->frame_num_offset += 256;

    }

    if (h->frame_num_offset == 0 || h->frame_num_offset >= h->prev_frame_num_offset) {

      av_log(h->s.avctx, AV_LOG_ERROR, "error in B-frame picture id\n");

      return -1;

    }

  } else {

    h->prev_frame_num = h->frame_num;

    h->frame_num = h->slice_num;

    h->prev_frame_num_offset = (h->frame_num - h->prev_frame_num);



    if (h->prev_frame_num_offset < 0) {

      h->prev_frame_num_offset += 256;

    }

  }



  for(m=0; m<2; m++){

    int i;

    for(i=0; i<4; i++){

      int j;

      for(j=-1; j<4; j++)

        h->ref_cache[m][scan8[0] + 8*i + j]= 1;

      h->ref_cache[m][scan8[0] + 8*i + j]= PART_NOT_AVAILABLE;

    }

  }



  for (s->mb_y=0; s->mb_y < s->mb_height; s->mb_y++) {

    for (s->mb_x=0; s->mb_x < s->mb_width; s->mb_x++) {



      if ( (get_bits_count(&s->gb) + 7) >= s->gb.size_in_bits &&

          ((get_bits_count(&s->gb) & 7) == 0 || show_bits (&s->gb, (-get_bits_count(&s->gb) & 7)) == 0)) {



        skip_bits(&s->gb, h->next_slice_index - get_bits_count(&s->gb));

        s->gb.size_in_bits = 8*buf_size;



        if (svq3_decode_slice_header (h))

          return -1;



        /* TODO: support s->mb_skip_run */

      }



      mb_type = svq3_get_ue_golomb (&s->gb);



      if (s->pict_type == I_TYPE) {

        mb_type += 8;

      } else if (s->pict_type == B_TYPE && mb_type >= 4) {

        mb_type += 4;

      }

      if (mb_type > 33 || svq3_decode_mb (h, mb_type)) {

        av_log(h->s.avctx, AV_LOG_ERROR, "error while decoding MB %d %d\n", s->mb_x, s->mb_y);

        return -1;

      }



      if (mb_type != 0) {

        hl_decode_mb (h);

      }



      if (s->pict_type != B_TYPE && !s->low_delay) {

        s->current_picture.mb_type[s->mb_x + s->mb_y*s->mb_stride] =

                        (s->pict_type == P_TYPE && mb_type < 8) ? (mb_type - 1) : -1;

      }

    }



    ff_draw_horiz_band(s, 16*s->mb_y, 16);

  }



  MPV_frame_end(s);



  if (s->pict_type == B_TYPE || s->low_delay) {

    *(AVFrame *) data = *(AVFrame *) &s->current_picture;

  } else {

    *(AVFrame *) data = *(AVFrame *) &s->last_picture;

  }



  avctx->frame_number = s->picture_number - 1;



  /* dont output the last pic after seeking */

  if (s->last_picture_ptr || s->low_delay) {

    *data_size = sizeof(AVFrame);

  }



  return buf_size;

}
