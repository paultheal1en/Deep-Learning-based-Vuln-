static int vp9_decode_update_thread_context(AVCodecContext *dst, const AVCodecContext *src)

{

    int i, res;

    VP9Context *s = dst->priv_data, *ssrc = src->priv_data;



    // FIXME scalability, size, etc.



    for (i = 0; i < 2; i++) {

        if (s->frames[i].tf.f->data[0])

            vp9_unref_frame(dst, &s->frames[i]);

        if (ssrc->frames[i].tf.f->data[0]) {

            if ((res = vp9_ref_frame(dst, &s->frames[i], &ssrc->frames[i])) < 0)

                return res;

        }

    }

    for (i = 0; i < 8; i++) {

        if (s->refs[i].f->data[0])

            ff_thread_release_buffer(dst, &s->refs[i]);

        if (ssrc->next_refs[i].f->data[0]) {

            if ((res = ff_thread_ref_frame(&s->refs[i], &ssrc->next_refs[i])) < 0)

                return res;

        }

    }



    s->invisible = ssrc->invisible;

    s->keyframe = ssrc->keyframe;

    s->uses_2pass = ssrc->uses_2pass;

    memcpy(&s->prob_ctx, &ssrc->prob_ctx, sizeof(s->prob_ctx));

    memcpy(&s->lf_delta, &ssrc->lf_delta, sizeof(s->lf_delta));

    if (ssrc->segmentation.enabled) {

        memcpy(&s->segmentation.feat, &ssrc->segmentation.feat,

               sizeof(s->segmentation.feat));

    }



    return 0;

}
