static int svq1_decode_frame(AVCodecContext *avctx,

                             void *data, int *data_size,

                             AVPacket *avpkt)

{

  const uint8_t *buf = avpkt->data;

  int buf_size = avpkt->size;

  MpegEncContext *s=avctx->priv_data;

  uint8_t        *current, *previous;

  int             result, i, x, y, width, height;

  AVFrame *pict = data;

  svq1_pmv *pmv;



  /* initialize bit buffer */

  init_get_bits(&s->gb,buf,buf_size*8);



  /* decode frame header */

  s->f_code = get_bits (&s->gb, 22);



  if ((s->f_code & ~0x70) || !(s->f_code & 0x60))

    return -1;



  /* swap some header bytes (why?) */

  if (s->f_code != 0x20) {

    uint32_t *src = (uint32_t *) (buf + 4);



    for (i=0; i < 4; i++) {

      src[i] = ((src[i] << 16) | (src[i] >> 16)) ^ src[7 - i];

    }

  }



  result = svq1_decode_frame_header (&s->gb, s);



  if (result != 0)

  {

    av_dlog(s->avctx, "Error in svq1_decode_frame_header %i\n",result);

    return result;

  }




  //FIXME this avoids some confusion for "B frames" without 2 references

  //this should be removed after libavcodec can handle more flexible picture types & ordering

  if(s->pict_type==AV_PICTURE_TYPE_B && s->last_picture_ptr==NULL) return buf_size;



  if(  (avctx->skip_frame >= AVDISCARD_NONREF && s->pict_type==AV_PICTURE_TYPE_B)

     ||(avctx->skip_frame >= AVDISCARD_NONKEY && s->pict_type!=AV_PICTURE_TYPE_I)

     || avctx->skip_frame >= AVDISCARD_ALL)

      return buf_size;



  if(MPV_frame_start(s, avctx) < 0)

      return -1;



  pmv = av_malloc((FFALIGN(s->width, 16)/8 + 3) * sizeof(*pmv));

  if (!pmv)

      return -1;



  /* decode y, u and v components */

  for (i=0; i < 3; i++) {

    int linesize;

    if (i == 0) {

      width  = FFALIGN(s->width, 16);

      height = FFALIGN(s->height, 16);

      linesize= s->linesize;

    } else {

      if(s->flags&CODEC_FLAG_GRAY) break;

      width  = FFALIGN(s->width/4, 16);

      height = FFALIGN(s->height/4, 16);

      linesize= s->uvlinesize;

    }



    current = s->current_picture.f.data[i];



    if(s->pict_type==AV_PICTURE_TYPE_B){

        previous = s->next_picture.f.data[i];

    }else{

        previous = s->last_picture.f.data[i];

    }



    if (s->pict_type == AV_PICTURE_TYPE_I) {

      /* keyframe */

      for (y=0; y < height; y+=16) {

        for (x=0; x < width; x+=16) {

          result = svq1_decode_block_intra (&s->gb, &current[x], linesize);

          if (result != 0)

          {

            av_log(s->avctx, AV_LOG_INFO, "Error in svq1_decode_block %i (keyframe)\n",result);

            goto err;

          }

        }

        current += 16*linesize;

      }

    } else {

      /* delta frame */

      memset (pmv, 0, ((width / 8) + 3) * sizeof(svq1_pmv));



      for (y=0; y < height; y+=16) {

        for (x=0; x < width; x+=16) {

          result = svq1_decode_delta_block (s, &s->gb, &current[x], previous,

                                            linesize, pmv, x, y);

          if (result != 0)

          {

            av_dlog(s->avctx, "Error in svq1_decode_delta_block %i\n",result);

            goto err;

          }

        }



        pmv[0].x =

        pmv[0].y = 0;



        current += 16*linesize;

      }

    }

  }



  *pict = *(AVFrame*)&s->current_picture;





  MPV_frame_end(s);



  *data_size=sizeof(AVFrame);

  result = buf_size;

err:

  av_free(pmv);

  return result;

}