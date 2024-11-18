static void print_formats(AVFilterContext *filter_ctx)

{

    int i, j;



#define PRINT_FMTS(inout, outin, INOUT)                                 \

    for (i = 0; i < filter_ctx->nb_##inout##puts; i++) {                     \

        if (filter_ctx->inout##puts[i]->type == AVMEDIA_TYPE_VIDEO) {   \

            AVFilterFormats *fmts =                                     \

                filter_ctx->inout##puts[i]->outin##_formats;            \

            for (j = 0; j < fmts->nb_formats; j++)                    \

                if(av_get_pix_fmt_name(fmts->formats[j]))               \

                printf(#INOUT "PUT[%d] %s: fmt:%s\n",                   \

                       i, filter_ctx->filter->inout##puts[i].name,      \

                       av_get_pix_fmt_name(fmts->formats[j]));          \

        } else if (filter_ctx->inout##puts[i]->type == AVMEDIA_TYPE_AUDIO) { \

            AVFilterFormats *fmts;                                      \

            AVFilterChannelLayouts *layouts;                            \

                                                                        \

            fmts = filter_ctx->inout##puts[i]->outin##_formats;         \

            for (j = 0; j < fmts->nb_formats; j++)                    \

                printf(#INOUT "PUT[%d] %s: fmt:%s\n",                   \

                       i, filter_ctx->filter->inout##puts[i].name,      \

                       av_get_sample_fmt_name(fmts->formats[j]));       \

                                                                        \

            layouts = filter_ctx->inout##puts[i]->outin##_channel_layouts; \

            for (j = 0; j < layouts->nb_channel_layouts; j++) {                  \

                char buf[256];                                          \

                av_get_channel_layout_string(buf, sizeof(buf), -1,      \

                                             layouts->channel_layouts[j]);         \

                printf(#INOUT "PUT[%d] %s: chlayout:%s\n",              \

                       i, filter_ctx->filter->inout##puts[i].name, buf); \

            }                                                           \

        }                                                               \

    }                                                                   \



    PRINT_FMTS(in,  out, IN);

    PRINT_FMTS(out, in,  OUT);

}
