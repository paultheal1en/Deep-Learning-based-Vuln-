static int rv34_decoder_alloc(RV34DecContext *r)

{

    r->intra_types_stride = r->s.mb_width * 4 + 4;



    r->cbp_chroma       = av_malloc(r->s.mb_stride * r->s.mb_height *

                                    sizeof(*r->cbp_chroma));

    r->cbp_luma         = av_malloc(r->s.mb_stride * r->s.mb_height *

                                    sizeof(*r->cbp_luma));

    r->deblock_coefs    = av_malloc(r->s.mb_stride * r->s.mb_height *

                                    sizeof(*r->deblock_coefs));

    r->intra_types_hist = av_malloc(r->intra_types_stride * 4 * 2 *

                                    sizeof(*r->intra_types_hist));

    r->mb_type          = av_mallocz(r->s.mb_stride * r->s.mb_height *

                                     sizeof(*r->mb_type));



    if (!(r->cbp_chroma       && r->cbp_luma && r->deblock_coefs &&

          r->intra_types_hist && r->mb_type)) {

        rv34_decoder_free(r);

        return AVERROR(ENOMEM);

    }



    r->intra_types = r->intra_types_hist + r->intra_types_stride * 4;



    return 0;

}
