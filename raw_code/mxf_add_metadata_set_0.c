static int mxf_add_metadata_set(MXFContext *mxf, void *metadata_set)

{

    int err;



    if (mxf->metadata_sets_count+1 >= UINT_MAX / sizeof(*mxf->metadata_sets))

        return AVERROR(ENOMEM);

    if ((err = av_reallocp_array(&mxf->metadata_sets, mxf->metadata_sets_count + 1,

                                 sizeof(*mxf->metadata_sets))) < 0) {

        mxf->metadata_sets_count = 0;

        return err;

    }

    mxf->metadata_sets[mxf->metadata_sets_count] = metadata_set;

    mxf->metadata_sets_count++;

    return 0;

}
