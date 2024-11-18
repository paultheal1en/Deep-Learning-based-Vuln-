static int parse_bsfs(void *log_ctx, const char *bsfs_spec,

                      AVBitStreamFilterContext **bsfs)

{

    char *bsf_name, *buf, *saveptr;

    int ret = 0;



    if (!(buf = av_strdup(bsfs_spec)))

        return AVERROR(ENOMEM);



    while (bsf_name = av_strtok(buf, ",", &saveptr)) {

        AVBitStreamFilterContext *bsf = av_bitstream_filter_init(bsf_name);



        if (!bsf) {

            av_log(log_ctx, AV_LOG_ERROR,

                   "Cannot initialize bitstream filter with name '%s', "

                   "unknown filter or internal error happened\n",

                   bsf_name);

            ret = AVERROR_UNKNOWN;

            goto end;

        }



        /* append bsf context to the list of bsf contexts */

        *bsfs = bsf;

        bsfs = &bsf->next;



        buf = NULL;

    }



end:

    av_free(buf);

    return ret;

}
