void ff_check_pixfmt_descriptors(void){

    int i, j;



    for (i=0; i<FF_ARRAY_ELEMS(av_pix_fmt_descriptors); i++) {

        const AVPixFmtDescriptor *d = &av_pix_fmt_descriptors[i];



        if (!d->name && !d->nb_components && !d->log2_chroma_w && !d->log2_chroma_h && !d->flags)

            continue;

//         av_log(NULL, AV_LOG_DEBUG, "Checking: %s\n", d->name);

        av_assert0(d->log2_chroma_w <= 3);

        av_assert0(d->log2_chroma_h <= 3);

        av_assert0(d->nb_components <= 4);

        av_assert0(d->name && d->name[0]);

        av_assert0((d->nb_components==4 || d->nb_components==2) == !!(d->flags & PIX_FMT_ALPHA));

        av_assert2(av_get_pix_fmt(d->name) == i);



        for (j=0; j<FF_ARRAY_ELEMS(d->comp); j++) {

            const AVComponentDescriptor *c = &d->comp[j];

            if(j>=d->nb_components)

                av_assert0(!c->plane && !c->step_minus1 && !c->offset_plus1 && !c->shift && !c->depth_minus1);

        }

    }

}
