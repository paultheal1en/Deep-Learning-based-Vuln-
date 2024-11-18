static void merge_context_after_encode(MpegEncContext *dst, MpegEncContext *src){

    int i;



    MERGE(dct_count[0]); //note, the other dct vars are not part of the context

    MERGE(dct_count[1]);

    MERGE(mv_bits);

    MERGE(i_tex_bits);

    MERGE(p_tex_bits);

    MERGE(i_count);

    MERGE(f_count);

    MERGE(b_count);

    MERGE(skip_count);

    MERGE(misc_bits);

    MERGE(er.error_count);

    MERGE(padding_bug_score);

    MERGE(current_picture.f->error[0]);

    MERGE(current_picture.f->error[1]);

    MERGE(current_picture.f->error[2]);



    if(dst->avctx->noise_reduction){

        for(i=0; i<64; i++){

            MERGE(dct_error_sum[0][i]);

            MERGE(dct_error_sum[1][i]);

        }

    }



    assert(put_bits_count(&src->pb) % 8 ==0);

    assert(put_bits_count(&dst->pb) % 8 ==0);

    avpriv_copy_bits(&dst->pb, src->pb.buf, put_bits_count(&src->pb));

    flush_put_bits(&dst->pb);

}
